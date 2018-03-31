#include "cnnetwork.h"
#include "helpers.h"

#include <CL/cl.hpp>

#include <algorithm>

namespace mathlib {

namespace {
constexpr char clprogram_src[] = R"(
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

__kernel void neuron_atom(__global const int3* nodes,
                          __global const double* inputs,
                          __global const double* weights,
                          __global const int* map,
                          __global double* inters,
                          __global double* outputs,
                          const int weights_stride) {
    const int neuron_id = get_global_id(0);
    const int weight_id = get_global_id(1);
    const int idx = neuron_id * weights_stride + weight_id;

    if (weight_id < nodes[neuron_id].y) {
        inters[idx] = inputs[map[idx]] * weights[idx + 1];
    } else {
        inters[idx] = 0;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = 2; i <= weights_stride; i *= 2) {
        if ((weight_id % i) != 0) {
            return;
        }

        inters[idx] += inters[idx + i / 2];

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // until here, only weight_id=0 can reach
    outputs[neuron_id] = sigmoid(inters[idx] + weights[idx]);  // taking into account the bias
}
)";
}

class cnnetwork::impl {
 public:
    impl(size_t inputs, std::vector<cnlayer>&& layers);

    size_t inputs_num() const {
        return inputs_;
    }
    size_t layer_num() const {
        return layers_.size();
    }
    const cnlayer& get_layer(size_t idx) const {
        return layers_[idx].desc;
    }

    std::vector<cl_double> weights(size_t layer, size_t neuron) const;
    void set_weights(size_t layer, size_t neuron, std::vector<cl_double> weights);
    cl_double bias(size_t layer, size_t neuron) const;
    void set_bias(size_t layer, size_t neuron, cl_double bias) const;

    std::vector<cl_double> exec(const std::vector<cl_double>& inputs);

 private:
    void make_input_layer(size_t inputs);
    void make_layer(cnlayer&& layer);

    struct cllayer {
        cnlayer desc;
        cl::Buffer const inputs;
        cl::Buffer nodes;  // neurons description
        const cl_int weights_stride;
        cl::Buffer weights;
        cl::Buffer mapping;
        cl::Buffer inter_outputs;
        cl::Buffer outputs;
    };

    using clnode = cl_int3;  // x -type, y -number of synapses, z -use bias

    const size_t inputs_;

    cl::Context context_ = cl::Context(CL_DEVICE_TYPE_GPU);
    cl::CommandQueue cmd_queue_;
    cl::Program prog_;
    cl::Kernel kernel_;
    cl::Buffer input_buf_;
    std::vector<cllayer> layers_;
};

cnnetwork::cnnetwork(size_t inputs, std::initializer_list<cnlayer>&& layers)
    : impl_(std::make_unique<cnnetwork::impl>(inputs, std::move(layers))) {
    if (impl_->layer_num() == 0) {
        throw std::logic_error("Neural network must have at least one layer");
    }
}

cnnetwork::~cnnetwork() = default;

size_t cnnetwork::inputs_num() const {
    return impl_->inputs_num();
}

size_t cnnetwork::layer_num() const {
    return impl_->layer_num();
}

cnlayer cnnetwork::layer_desc(size_t idx) const {
    return impl_->get_layer(idx);
}

std::vector<double> cnnetwork::weights(size_t layer, size_t neuron) const {
    return impl_->weights(layer, neuron);
}

void cnnetwork::set_weights(size_t layer, size_t neuron, std::vector<double> weights) {
    impl_->set_weights(layer, neuron, weights);
}

double cnnetwork::bias(size_t layer, size_t neuron) const {
    return impl_->bias(layer, neuron);
}

void cnnetwork::set_bias(size_t layer, size_t neuron, double bias) const {
    impl_->set_bias(layer, neuron, bias);
}

std::vector<double> cnnetwork::operator()(const std::vector<double>& inputs) {
    static_assert(std::is_same<double, cl_double>::value, "OpenCL double type does not match system double type.");
    return impl_->exec(inputs);
}

cnnetwork::impl::impl(size_t inputs, std::vector<cnlayer>&& layers) : inputs_(inputs) {
    const auto devs = context_.getInfo<CL_CONTEXT_DEVICES>();
    if (!devs.empty()) {
        const auto& dev = devs.front();
        if (!(dev.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU) || !dev.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>()) {
            throw std::logic_error("OpenCL device is unsuitable");
        }
    } else {
        throw std::logic_error("OpenCL GPU-device has not been found");
    }
    cl::CommandQueue qcmd(context_, devs.front());
    std::swap(cmd_queue_, qcmd);

    make_input_layer(inputs);
    for (auto&& l : layers) {
        make_layer(std::move(l));
    }

    cl_int err;
    cl::Program prog(context_, clprogram_src, true, &err);
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL programm has not been built with error code: " + std::to_string(err));
    }
    std::swap(prog_, prog);
    cl::Kernel kernel(prog_, "neuron_atom", &err);
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL kernel has not been created with error code: " + std::to_string(err));
    }
    std::swap(kernel_, kernel);
}

std::vector<cl_double> cnnetwork::impl::weights(size_t l, size_t n) const {
    const auto& layer = layers_[l];
    std::vector<cl_double> buf(layer.desc[n].neuron.synapses);
    auto err = cmd_queue_.enqueueReadBuffer(layer.weights, CL_TRUE, (n * layer.weights_stride + 1) * sizeof(cl_double),
                                            buf.size() * sizeof(cl_double), buf.data());
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL weights buffer has not been read with error code: " + std::to_string(err));
    }
    return buf;
}

void cnnetwork::impl::set_weights(size_t l, size_t n, std::vector<cl_double> w) {
    const auto& layer = layers_[l];
    auto err = cmd_queue_.enqueueWriteBuffer(layer.weights, CL_TRUE, (n * layer.weights_stride + 1) * sizeof(cl_double),
                                             w.size() * sizeof(cl_double), w.data());
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL weights buffer has not been written with error code: " + std::to_string(err));
    }
}

cl_double cnnetwork::impl::bias(size_t l, size_t n) const {
    const auto& layer = layers_[l];
    cl_double buf{};
    auto err = cmd_queue_.enqueueReadBuffer(layer.weights, CL_TRUE, (n * layer.weights_stride) * sizeof(cl_double),
                                            sizeof(cl_double), &buf);
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL weights buffer has not been read with error code: " + std::to_string(err));
    }
    return buf;
}

void cnnetwork::impl::set_bias(size_t l, size_t n, cl_double b) const {
    const auto& layer = layers_[l];
    auto err = cmd_queue_.enqueueWriteBuffer(layer.weights, CL_TRUE, (n * layer.weights_stride) * sizeof(cl_double),
                                             sizeof(cl_double), &b);
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL weights buffer has not been written with error code: " + std::to_string(err));
    }
}

std::vector<cl_double> cnnetwork::impl::exec(const std::vector<cl_double>& inputs) {
    {
        auto err =
            cmd_queue_.enqueueWriteBuffer(input_buf_, CL_TRUE, 0, sizeof(cl_double) * inputs.size(), inputs.data());
        if (err != CL_SUCCESS) {
            throw std::logic_error("OpenCL inputs buffer has not been written with error code: " + std::to_string(err));
        }
    }
    {
        for (auto& layer : layers_) {
            const cl::NDRange range(layer.desc.size(), layer.weights_stride);
            kernel_.setArg(0, layer.nodes);
            kernel_.setArg(1, layer.inputs);
            kernel_.setArg(2, layer.weights);
            kernel_.setArg(3, layer.mapping);
            kernel_.setArg(4, layer.inter_outputs);
            kernel_.setArg(5, layer.outputs);
            kernel_.setArg(6, layer.weights_stride);
            auto err = cmd_queue_.enqueueNDRangeKernel(kernel_, cl::NDRange(0, 0), range);
            if (err != CL_SUCCESS) {
                throw std::logic_error("OpenCL kernel has not been executed with error code: " + std::to_string(err));
            }
        }
    }
    std::vector<cl_double> result(layers_.back().desc.size());
    {
        auto err = cmd_queue_.enqueueReadBuffer(layers_.back().outputs, CL_TRUE, 0,
                                                sizeof(cl_double) * layers_.back().desc.size(), result.data());
        if (err != CL_SUCCESS) {
            throw std::logic_error("OpenCL outputs buffer has not been read with error code: " + std::to_string(err));
        }
    }
    return result;
}

void cnnetwork::impl::make_input_layer(size_t inputs) {
    cl::Buffer buf(context_, CL_MEM_READ_ONLY, sizeof(cl_double) * inputs);
    std::swap(input_buf_, buf);
}

void cnnetwork::impl::make_layer(cnlayer&& layer) {
    const size_t layer_size = layer.size();
    const size_t input_size = layers_.empty() ? inputs_ : layers_.back().desc.size();
    {
        // Finding maximum number of synapses including the bias
        size_t max_w = 0;
        for (const auto& n : layer) {
            if (!n.neuron.synapses) {
                throw std::logic_error("There should be synapses");
            }
            max_w = std::max(max_w, n.neuron.synapses + 1);  // the bias is counted always
        }
        // Creating all OpenCL objects that are necessary for presentation of the layer.
        const size_t w_stride = nearest_upper_pow2(max_w);
        layers_.emplace_back(cllayer{std::move(layer), layers_.empty() ? input_buf_ : layers_.back().outputs,
                                     cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(clnode) * layer_size),
                                     static_cast<cl_int>(w_stride),
                                     cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(cl_double) * w_stride * layer_size),
                                     cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(cl_int) * w_stride * layer_size),
                                     cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(cl_double) * w_stride * layer_size),
                                     cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(cl_double) * layer_size)});
    }
    {
        // Initializing layer nodes data.
        cllayer& curr_cllayer = layers_.back();
        std::vector<clnode> clnodes(layer_size, clnode{0});
        std::vector<cl_double> clweights(curr_cllayer.weights_stride * layer_size, cl_double{0});
        std::vector<cl_int> clmap(curr_cllayer.weights_stride * layer_size, cl_int{0});
        for (size_t i = 0; i < layer_size; ++i) {
            const cnnode& node = curr_cllayer.desc[i];
            clnodes[i].x = static_cast<cl_int>(node.neuron.type);
            clnodes[i].y = static_cast<cl_int>(node.neuron.synapses);
            clnodes[i].z = node.neuron.bias ? CL_TRUE : CL_FALSE;
            if (node.map.size() != node.neuron.synapses) {
                throw std::logic_error("Mapping size must be equal number of synapses");
            }
            const size_t offset = i * curr_cllayer.weights_stride;
            for (size_t w = 0; w < node.neuron.synapses; ++w) {
                clweights[offset + 1 + w] = 1;  // the bias is the first always and zero by default
                if (node.map[w] < 0 || node.map[w] >= input_size) {
                    throw std::logic_error("Mapping index must be reference to one of inputs");
                }
                clmap[offset + w] = node.map[w];
            }
        }
        {
            auto err = cmd_queue_.enqueueWriteBuffer(curr_cllayer.nodes, CL_TRUE, 0, sizeof(clnode) * clnodes.size(),
                                                     clnodes.data());
            if (err != CL_SUCCESS) {
                throw std::logic_error("OpenCL nodes buffer has not been written with error code: " +
                                       std::to_string(err));
            }
        }
        {
            auto err = cmd_queue_.enqueueWriteBuffer(curr_cllayer.weights, CL_TRUE, 0,
                                                     sizeof(cl_double) * clweights.size(), clweights.data());
            if (err != CL_SUCCESS) {
                throw std::logic_error("OpenCL weights buffer has not been written with error code: " +
                                       std::to_string(err));
            }
        }
        {
            auto err = cmd_queue_.enqueueWriteBuffer(curr_cllayer.mapping, CL_TRUE, 0, sizeof(cl_int) * clmap.size(),
                                                     clmap.data());
            if (err != CL_SUCCESS) {
                throw std::logic_error("OpenCL mapping buffer has not been written with error code: " +
                                       std::to_string(err));
            }
        }
    }
}

}  // namespace mathlib
