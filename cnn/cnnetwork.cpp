#include "cnnetwork.h"
#include "clprogram.h"
#include "math/mathlib/helpers.h"

#include <CL/cl.hpp>

#include <algorithm>
#include <sstream>

namespace cnn {

class cnnetwork::impl {
 public:
    impl(size_t inputs, const std::vector<cnlayer>& layers);

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
    void set_weights(size_t layer, size_t neuron, const std::vector<cl_double>& weights);

    std::vector<cl_double> exec(const std::vector<cl_double>& inputs);

 private:
    size_t weights_number(size_t l) const;
    size_t make_input_layer(size_t inputs, bool bias);
    size_t make_layer(size_t input_size, cnlayer layer, bool bias);

    struct cllayer {
        cnlayer desc;
        cl_int stride;             // size of input buffer
        cl::Buffer inputs;         // reference to exist buffer
        cl::Buffer weights;        // weights of all nodes
        cl::Buffer inter_outputs;  // the same size like weights buffer
        cl::Buffer outputs;        // result of all nodes plus 1.0 for bias of next layer if it is present
    };

    const size_t inputs_;

    cl::Context context_ = cl::Context(CL_DEVICE_TYPE_GPU);
    cl::CommandQueue cmd_queue_;
    cl::Program prog_;
    cl::Kernel kernel_;
    cl::Buffer input_buf_;
    std::vector<cllayer> layers_;
};

cnnetwork::cnnetwork(size_t inputs, const std::vector<cnlayer>& layers)
    : impl_(std::make_unique<cnnetwork::impl>(inputs, layers)) {
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

void cnnetwork::set_weights(size_t layer, size_t neuron, const std::vector<double>& weights) {
    impl_->set_weights(layer, neuron, weights);
}

std::vector<double> cnnetwork::operator()(const std::vector<double>& inputs) {
    static_assert(std::is_same<double, cl_double>::value, "OpenCL double type does not match system double type.");
    return impl_->exec(inputs);
}

cnnetwork::impl::impl(size_t inputs, const std::vector<cnlayer>& layers) : inputs_(inputs) {
    if (layers.empty()) {
        return;
    }

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

    auto stride = make_input_layer(inputs, layers[0].bias);
    for (size_t i = 0; i < (layers.size() - 1); ++i) {
        stride = make_layer(stride, layers[i], layers[i + 1].bias);
    }
    make_layer(stride, layers.back(), false);

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
    const auto weights_num = weights_number(l);
    std::vector<cl_double> buf(weights_num);
    auto err = cmd_queue_.enqueueReadBuffer(layer.weights, CL_TRUE, (n * layer.stride) * sizeof(cl_double),
                                            weights_num * sizeof(cl_double), buf.data());
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL weights buffer has not been read with error code: " + std::to_string(err));
    }
    return buf;
}

void cnnetwork::impl::set_weights(size_t l, size_t n, const std::vector<cl_double>& w) {
    const auto& layer = layers_[l];
    const auto weights_num = weights_number(l);
    if (w.size() != weights_num) {
        std::stringstream str;
        str << "Given " << w.size() << " weights, expected " << weights_num;
        if (layer.desc.bias) {
            str << " including bias";
        }
        throw std::logic_error(str.str());
    }
    auto err = cmd_queue_.enqueueWriteBuffer(layer.weights, CL_TRUE, (n * layer.stride) * sizeof(cl_double),
                                             weights_num * sizeof(cl_double), w.data());
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL weights buffer has not been written with error code: " + std::to_string(err));
    }
}

std::vector<cl_double> cnnetwork::impl::exec(const std::vector<cl_double>& inputs) {
    if (inputs.size() != inputs_) {
        std::stringstream str;
        str << "Given " << inputs.size() << " input values, expected " << inputs_;
        throw std::logic_error(str.str());
    }
    {
        auto err = cmd_queue_.enqueueWriteBuffer(input_buf_, CL_TRUE, 0, sizeof(cl_double) * inputs_, inputs.data());
        if (err != CL_SUCCESS) {
            throw std::logic_error("OpenCL inputs buffer has not been written with error code: " + std::to_string(err));
        }
    }
    {
        for (auto& layer : layers_) {
            const cl::NDRange range(layer.desc.nodes, layer.stride);
            kernel_.setArg(0, layer.stride);
            kernel_.setArg(1, layer.inputs);
            kernel_.setArg(2, layer.weights);
            kernel_.setArg(3, layer.inter_outputs);
            kernel_.setArg(4, layer.outputs);
            auto err = cmd_queue_.enqueueNDRangeKernel(kernel_, cl::NDRange(0, 0), range);
            if (err != CL_SUCCESS) {
                throw std::logic_error("OpenCL kernel has not been executed with error code: " + std::to_string(err));
            }
        }
    }
    std::vector<cl_double> result(layers_.back().desc.nodes);
    {
        auto err = cmd_queue_.enqueueReadBuffer(layers_.back().outputs, CL_TRUE, 0, sizeof(cl_double) * result.size(),
                                                result.data());
        if (err != CL_SUCCESS) {
            throw std::logic_error("OpenCL outputs buffer has not been read with error code: " + std::to_string(err));
        }
    }
    return result;
}

size_t cnnetwork::impl::weights_number(size_t l) const {
    return (l == 0 ? inputs_ : layers_[l - 1].desc.nodes) + (layers_[l].desc.bias ? 1 : 0);
}

size_t cnnetwork::impl::make_input_layer(size_t inputs, bool bias) {
    const size_t stride = mathlib::nearest_upper_pow2(inputs + (bias ? 1 : 0));
    cl::Buffer buf(context_, CL_MEM_READ_ONLY, sizeof(cl_double) * stride);
    std::swap(input_buf_, buf);
    std::vector<cl_double> init_data(stride, cl_double{0});
    if (bias) {
        init_data[inputs] = 1.0;
    }
    auto err = cmd_queue_.enqueueWriteBuffer(input_buf_, CL_TRUE, 0, sizeof(cl_double) * stride, init_data.data());
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL nodes buffer has not been written with error code: " + std::to_string(err));
    }
    return stride;
}

size_t cnnetwork::impl::make_layer(size_t input_size, cnlayer layer, bool bias) {
    const size_t layer_size = layer.nodes;
    const size_t output_stride = mathlib::nearest_upper_pow2(layer.nodes + (bias ? 1 : 0));
    // Creating all OpenCL objects that are necessary for presentation of the layer.
    layers_.push_back(cllayer{
        std::move(layer),                                       // desc
        static_cast<cl_int>(input_size),                        // stride of this layer
        layers_.empty() ? input_buf_ : layers_.back().outputs,  // reference to input buffer
        cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(cl_double) * input_size * layer_size),   // weights buffer
        cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(cl_double) * input_size * layer_size),  // intermediate buffer
        cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(cl_double) * output_stride)});          // output buffer
    {
        // Initializing layer data.
        auto& curr_cllayer = layers_.back();
        std::vector<cl_double> init_weights(input_size * layer_size, cl_double{0});
        {
            auto err = cmd_queue_.enqueueWriteBuffer(curr_cllayer.weights, CL_TRUE, 0,
                                                     sizeof(cl_double) * init_weights.size(), init_weights.data());
            if (err != CL_SUCCESS) {
                throw std::logic_error("OpenCL weights buffer has not been written with error code: " +
                                       std::to_string(err));
            }
        }
        {
            auto err = cmd_queue_.enqueueWriteBuffer(curr_cllayer.inter_outputs, CL_TRUE, 0,
                                                     sizeof(cl_double) * init_weights.size(), init_weights.data());
            if (err != CL_SUCCESS) {
                throw std::logic_error("OpenCL weights buffer has not been written with error code: " +
                                       std::to_string(err));
            }
        }
        std::vector<cl_double> init_outputs(output_stride, cl_double{0});
        if (bias) {
            init_outputs[layer_size] = 1.0;
        }
        {
            auto err = cmd_queue_.enqueueWriteBuffer(curr_cllayer.outputs, CL_TRUE, 0,
                                                     sizeof(cl_double) * init_outputs.size(), init_outputs.data());
            if (err != CL_SUCCESS) {
                throw std::logic_error("OpenCL weights buffer has not been written with error code: " +
                                       std::to_string(err));
            }
        }
    }
    return output_stride;
}

}  // namespace cnn
