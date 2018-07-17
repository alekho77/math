#include "cntrainer_impl.h"

#include <random>
#include <algorithm>
#include <sstream>

namespace cnn {

cntrainer_impl::cntrainer_impl(cnnetwork_impl& network) : network_(network) {
    for (const auto& l : network_.layers_) {
        layers_.push_back(train_layer{
            l,  // reference to all data of network layer
            cl::Buffer(network_.context_, CL_MEM_READ_WRITE, sizeof(cl_double) * l.output_size),  // deltas buffer
            cl::Buffer(network_.context_, CL_MEM_READ_WRITE,
                       sizeof(cl_double) * l.input_size * l.output_size)  // previous adjustments buffer
        });
        // Initializing layer data
        auto& curr_cllayer = layers_.back();
        {
            std::vector<cl_double> init_deltas(l.output_size, cl_double{0});
            auto err = network_.cmd_queue_.enqueueWriteBuffer(
                curr_cllayer.deltas, CL_TRUE, 0, sizeof(cl_double) * init_deltas.size(), init_deltas.data());
            if (err != CL_SUCCESS) {
                throw std::logic_error("OpenCL weights buffer has not been written with error code: " +
                                       std::to_string(err));
            }
        }
        {
            std::vector<cl_double> init_adjusts(l.input_size * l.output_size, cl_double{0});
            auto err = network_.cmd_queue_.enqueueWriteBuffer(
                curr_cllayer.adjustments, CL_TRUE, 0, sizeof(cl_double) * init_adjusts.size(), init_adjusts.data());
            if (err != CL_SUCCESS) {
                throw std::logic_error("OpenCL weights buffer has not been written with error code: " +
                                       std::to_string(err));
            }
        }
    }
    cl_int err{};
    delta_kernel_ = cl::Kernel(network_.prog_, "delta_atom", &err);
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL kernel has not been created with error code: " + std::to_string(err));
    }
    adjust_kernel_ = cl::Kernel(network_.prog_, "adjust_atom", &err);
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL kernel has not been created with error code: " + std::to_string(err));
    }
}

void cntrainer_impl::randomize(cl_double range, unsigned seed) {
    // Since there is no native OpenCL rand function it seems we should do it on CPU.
    std::mt19937 gen(seed);
    std::uniform_real_distribution<cl_double> dis(-range, range);
    for (size_t l = 0; l < network_.layer_num(); ++l) {
        const auto nodes = network_.get_layer(l).nodes;
        const auto weights_number = network_.weights_number(l);
        for (size_t n = 0; n < nodes; ++n) {
            std::vector<cl_double> weights(weights_number);
            std::generate(weights.begin(), weights.end(), [&gen, &dis]() { return dis(gen); });
            network_.set_weights(l, n, weights);
        }
    }
}

void cntrainer_impl::reset() {
    for (auto& layer : layers_) {
        std::vector<cl_double> init_adjusts(layer.network_layer.input_size * layer.network_layer.output_size,
                                            cl_double{0});
        auto err = network_.cmd_queue_.enqueueWriteBuffer(layer.adjustments, CL_TRUE, 0,
                                                          sizeof(cl_double) * init_adjusts.size(), init_adjusts.data());
        if (err != CL_SUCCESS) {
            throw std::logic_error("OpenCL weights buffer has not been written with error code: " +
                                   std::to_string(err));
        }
    }
}

std::tuple<cl_double, cl_double> cntrainer_impl::exec(const std::vector<cl_double>& inputs,
                                                      const std::vector<cl_double>& desired_outputs) {
    // Forward pass to determine values of all neurons.
    const auto actual_outputs = network_.exec(inputs);
    // Back pass to get deltas
    if (actual_outputs.size() != desired_outputs.size()) {
        std::stringstream str;
        str << "Given " << desired_outputs.size() << " sample values, expected " << actual_outputs.size();
        throw std::logic_error(str.str());
    }
    {
        // Actually output size is quite small so we feel free to make output deltas on CPU
        std::vector<cl_double> output_deltas;
        output_deltas.reserve(actual_outputs.size());
        for (size_t i = 0; i < actual_outputs.size(); i++) {
            output_deltas.push_back((desired_outputs[i] - actual_outputs[i]) * actual_outputs[i] *
                                    (1 - actual_outputs[i]));
        }
        compute_deltas(output_deltas);
    }
    // Second forward pass to modify the weights
    adjust_weights();
    // Third forward pass to compute adjusted result
    const auto adjusted_outputs = network_.exec(inputs);
    {
        // Again, output size is quite small so we perform to compute errors on CPU
    }
    return std::tuple<cl_double, cl_double>();
}

cl_double cntrainer_impl::learning_rate() const {
    return eta_;
}

void cntrainer_impl::set_learning_rate(cl_double eta) {
    eta_ = eta;
}

cl_double cntrainer_impl::momentum() const {
    return alpha_;
}

void cntrainer_impl::set_momentum(cl_double alpha) {
    alpha_ = alpha;
}

std::vector<cl_double> cntrainer_impl::states(size_t l) const {
    const auto& layer = layers_[l].network_layer;
    std::vector<cl_double> result(layer.desc.nodes);
    auto err = network_.cmd_queue_.enqueueReadBuffer(layer.outputs, CL_TRUE, 0, sizeof(cl_double) * result.size(),
                                                     result.data());
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL outputs buffer has not been read with error code: " + std::to_string(err));
    }
    return result;
}

std::vector<cl_double> cntrainer_impl::deltas(size_t l) const {
    const auto& layer = layers_[l];
    std::vector<cl_double> result(layer.network_layer.desc.nodes);
    auto err = network_.cmd_queue_.enqueueReadBuffer(layer.deltas, CL_TRUE, 0, sizeof(cl_double) * result.size(),
                                                     result.data());
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL outputs buffer has not been read with error code: " + std::to_string(err));
    }
    return result;
}

void cntrainer_impl::compute_deltas(const std::vector<cl_double>& output_deltas) {
    auto layer = layers_.rbegin();
    {
        auto err = network_.cmd_queue_.enqueueWriteBuffer(
            layer->deltas, CL_TRUE, 0, sizeof(cl_double) * output_deltas.size(), output_deltas.data());
        if (err != CL_SUCCESS) {
            throw std::logic_error("OpenCL inputs buffer has not been written with error code: " + std::to_string(err));
        }
    }
    for (++layer; layer != layers_.rend(); ++layer) {
        auto prev_layer = layer - 1;
        const cl::NDRange range(prev_layer->network_layer.output_size, prev_layer->network_layer.input_size);
        delta_kernel_.setArg(0, prev_layer->network_layer.weights);
        delta_kernel_.setArg(1, prev_layer->network_layer.inter_outputs);
        delta_kernel_.setArg(2, prev_layer->deltas);
        delta_kernel_.setArg(3, layer->network_layer.outputs);
        delta_kernel_.setArg(4, layer->deltas);
        auto err = network_.cmd_queue_.enqueueNDRangeKernel(delta_kernel_, cl::NDRange(0, 0), range);
        if (err != CL_SUCCESS) {
            throw std::logic_error("OpenCL kernel has not been executed with error code: " + std::to_string(err));
        }
    }
}

void cntrainer_impl::adjust_weights() {
    for (const auto& layer : layers_) {
        const cl::NDRange range(layer.network_layer.output_size, layer.network_layer.input_size);
        adjust_kernel_.setArg(0, eta_);
        adjust_kernel_.setArg(1, alpha_);
        adjust_kernel_.setArg(2, layer.network_layer.inputs);
        adjust_kernel_.setArg(3, layer.deltas);
        adjust_kernel_.setArg(4, layer.network_layer.weights);
        adjust_kernel_.setArg(5, layer.adjustments);
        auto err = network_.cmd_queue_.enqueueNDRangeKernel(adjust_kernel_, cl::NDRange(0, 0), range);
        if (err != CL_SUCCESS) {
            throw std::logic_error("OpenCL kernel has not been executed with error code: " + std::to_string(err));
        }
    }
}

}  // namespace cnn
