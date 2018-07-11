#include "cnnetwork_impl.h"
#include "clprogram.h"
#include "math/mathlib/helpers.h"

#include <sstream>

namespace cnn {

cnnetwork_impl::cnnetwork_impl(size_t inputs, const std::vector<cnlayer>& layers) : inputs_(inputs) {
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
    cmd_queue_ = cl::CommandQueue(context_, devs.front());

    auto stride = make_input_layer(inputs, layers[0].bias);
    for (size_t i = 0; i < (layers.size() - 1); ++i) {
        stride = make_layer(stride, layers[i], layers[i + 1].bias);
    }
    make_layer(stride, layers.back(), false);

    cl_int err;
    prog_ = cl::Program(context_, clprogram_src, true, &err);
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL programm has not been built with error code: " + std::to_string(err));
    }
    kernel_ = cl::Kernel(prog_, "neuron_atom", &err);
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL kernel has not been created with error code: " + std::to_string(err));
    }
}

std::vector<cl_double> cnnetwork_impl::weights(size_t l, size_t n) const {
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

void cnnetwork_impl::set_weights(size_t l, size_t n, const std::vector<cl_double>& w) {
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

std::vector<cl_double> cnnetwork_impl::exec(const std::vector<cl_double>& inputs) {
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

size_t cnnetwork_impl::weights_number(size_t l) const {
    return (l == 0 ? inputs_ : layers_[l - 1].desc.nodes) + (layers_[l].desc.bias ? 1 : 0);
}

size_t cnnetwork_impl::make_input_layer(size_t inputs, bool bias) {
    const size_t stride = mathlib::nearest_upper_pow2(inputs + (bias ? 1 : 0));
    input_buf_ = cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(cl_double) * stride);
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

size_t cnnetwork_impl::make_layer(size_t input_size, cnlayer layer, bool bias) {
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
