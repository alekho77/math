#include "cntrainer_impl.h"

#include <random>
#include <algorithm>

namespace cnn {

cntrainer_impl::cntrainer_impl(cnnetwork_impl& network) : network_(network) {}

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
                                                      const std::vector<cl_double>& /*desired_outputs*/) {
    // Forward pass to determine values of all neurons.
    network_.exec(inputs);
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

}  // namespace cnn
