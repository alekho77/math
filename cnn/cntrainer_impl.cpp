#include "cntrainer_impl.h"

#include <random>
#include <algorithm>
#include <sstream>

namespace cnn {

cntrainer_impl::cntrainer_impl(cnnetwork_impl& network) : network_(network) {
    samples_buf_ =
        cl::Buffer(network_.context_, CL_MEM_READ_ONLY, sizeof(cl_double) * network_.layers_.back().desc.nodes);
    for (const auto& l : network_.layers_) {
        layers_.push_back(
            train_layer{cl::Buffer(network_.context_, CL_MEM_READ_WRITE, sizeof(cl_double) * l.desc.nodes)});
        // Initializing layer data.
        auto& curr_cllayer = layers_.back();
        std::vector<cl_double> init_deltas(l.desc.nodes, cl_double{0});
        {
            auto err = network_.cmd_queue_.enqueueWriteBuffer(
                curr_cllayer.deltas, CL_TRUE, 0, sizeof(cl_double) * init_deltas.size(), init_deltas.data());
            if (err != CL_SUCCESS) {
                throw std::logic_error("OpenCL weights buffer has not been written with error code: " +
                                       std::to_string(err));
            }
        }
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

std::tuple<cl_double, cl_double> cntrainer_impl::exec(const std::vector<cl_double>& inputs,
                                                      const std::vector<cl_double>& desired_outputs) {
    // Forward pass to determine values of all neurons.
    const auto actual_outputs = network_.exec(inputs);
    // First back pass to get deltas
    compute_deltas(desired_outputs);
    return std::tuple<cl_double, cl_double>();
}

std::vector<cl_double> cntrainer_impl::states(size_t l) const {
    const auto layer = network_.layers_[l];
    std::vector<cl_double> result(layer.desc.nodes);
    auto err = network_.cmd_queue_.enqueueReadBuffer(layer.outputs, CL_TRUE, 0, sizeof(cl_double) * result.size(),
                                                     result.data());
    if (err != CL_SUCCESS) {
        throw std::logic_error("OpenCL outputs buffer has not been read with error code: " + std::to_string(err));
    }
    return result;
}

void cntrainer_impl::compute_deltas(const std::vector<cl_double>& desired_outputs) {
    if (desired_outputs.size() != network_.layers_.back().desc.nodes) {
        std::stringstream str;
        str << "Given " << desired_outputs.size() << " sample values, expected " << network_.layers_.back().desc.nodes;
        throw std::logic_error(str.str());
    }
    {
        auto err = network_.cmd_queue_.enqueueWriteBuffer(
            samples_buf_, CL_TRUE, 0, sizeof(cl_double) * desired_outputs.size(), desired_outputs.data());
        if (err != CL_SUCCESS) {
            throw std::logic_error("OpenCL inputs buffer has not been written with error code: " + std::to_string(err));
        }
    }
    {
        auto layer = layers_.rbegin();
        for (++layer; layer != layers_.rend(); ++layer) {
            // const cl::NDRange range(layer.desc.nodes, layer.stride);
            // kernel_.setArg(0, layer.stride);
            // kernel_.setArg(1, layer.inputs);
            // kernel_.setArg(2, layer.weights);
            // kernel_.setArg(3, layer.inter_outputs);
            // kernel_.setArg(4, layer.outputs);
            // auto err = cmd_queue_.enqueueNDRangeKernel(kernel_, cl::NDRange(0, 0), range);
            // if (err != CL_SUCCESS) {
            //    throw std::logic_error("OpenCL kernel has not been executed with error code: " + std::to_string(err));
            //}
        }
    }
}

}  // namespace cnn
