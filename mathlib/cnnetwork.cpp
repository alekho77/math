#include "cnnetwork.h"
#include "helpers.h"

#include <CL/cl.hpp>

#include <algorithm>

namespace mathlib {

namespace {
constexpr char clprogram_src[] = R"(
__kernel void neuron_atom(__global int3* nodes,
                          __global double* inputs,
                          __global double* weights,
                          __global int* map,
                          __global double* inters,
                          __global double* outputs) {
  size_t neuron_id = get_global_id(0);
  size_t weight_id = get_global_id(1);
}
)";
}

class cnnetwork::impl {
 public:
    impl(size_t inputs, std::initializer_list<cnlayer>&& layers);

    size_t inputs_num() const {
        return inputs_;
    }
    size_t layer_num() const {
        return layers_desc_.size();
    }
    const cnlayer& get_layer(size_t idx) const {
        return layers_desc_[idx];
    }

 private:
    void make_input_layer(size_t inputs);
    void make_layer(const cnlayer& layer);

    struct cllayer {
        cl::Buffer const& inputs;
        cl::Buffer nodes;  // neurons description
        const size_t weights_stride;
        cl::Buffer weights;
        cl::Buffer mapping;
        cl::Buffer inter_outputs;
        cl::Buffer outputs;
    };

    using clnode = cl_int3;  // x -type, y -number of synapses, z -use bias

    const size_t inputs_;
    const std::vector<cnlayer> layers_desc_;

    cl::Context context_ = cl::Context(CL_DEVICE_TYPE_GPU);
    cl::CommandQueue cmd_queue_;
    cl::Buffer input_buf_;
    std::vector<cllayer> layers_;
};

cnnetwork::cnnetwork(size_t inputs, std::initializer_list<cnlayer>&& layers)
    : impl_(std::make_unique<cnnetwork::impl>(inputs, std::move(layers))) {}

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

std::vector<double> cnnetwork::operator()(const std::vector<double>& /*inputs*/) {
    return std::vector<double>();
}

cnnetwork::impl::impl(size_t inputs, std::initializer_list<cnlayer>&& layers)
    : inputs_(inputs), layers_desc_(std::move(layers)) {
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
    for (const auto& l : layers_desc_) {
        make_layer(l);
    }

    // const std::string src{clprogram_src};
    // cl::Program prog(src);
}

void cnnetwork::impl::make_input_layer(size_t inputs) {
    cl::Buffer buf(context_, CL_MEM_READ_ONLY, sizeof(cl_double) * inputs);
    std::swap(input_buf_, buf);
}

void cnnetwork::impl::make_layer(const cnlayer& layer) {
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
        layers_.emplace_back(
            cllayer{layers_.empty() ? input_buf_ : layers_.back().outputs,
                    cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(clnode) * layer.size()), w_stride,
                    cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(cl_double) * w_stride * layer.size()),
                    cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(cl_int) * w_stride * layer.size()),
                    cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(cl_double) * w_stride * layer.size()),
                    cl::Buffer(context_, CL_MEM_READ_WRITE, sizeof(cl_double) * layer.size())});
    }
    {
        // Initializing layer nodes data.
        cllayer& curr_cllayer = layers_.back();
        std::vector<clnode> clnodes(layer.size(), clnode{0});
        std::vector<cl_double> clweights(curr_cllayer.weights_stride * layer.size(), cl_double{0});
        std::vector<cl_int> clmap(curr_cllayer.weights_stride * layer.size(), cl_int{0});
        for (size_t i = 0; i < layer.size(); ++i) {
            const cnnode& node = layer[i];
            clnodes[i].x = static_cast<cl_int>(node.neuron.type);
            clnodes[i].y = static_cast<cl_int>(node.neuron.synapses);
            clnodes[i].z = node.neuron.bias ? CL_TRUE : CL_FALSE;
            if (node.map.size() != node.neuron.synapses) {
                throw std::logic_error("Mapping size must be equal number of synapses");
            }
            const size_t offset = i * curr_cllayer.weights_stride;
            for (size_t w = 0; w < node.neuron.synapses; ++w) {
                clweights[offset + 1 + w] = 1;  // the bias is the first always and zero by default
                clmap[offset + w] = node.map[w];
            }
        }
        cmd_queue_.enqueueWriteBuffer(curr_cllayer.nodes, CL_TRUE, 0, sizeof(clnode) * clnodes.size(), clnodes.data());
        cmd_queue_.enqueueWriteBuffer(curr_cllayer.weights, CL_TRUE, 0, sizeof(cl_double) * clweights.size(),
                                      clweights.data());
        cmd_queue_.enqueueWriteBuffer(curr_cllayer.mapping, CL_TRUE, 0, sizeof(cl_int) * clmap.size(), clmap.data());
    }
}

}  // namespace mathlib
