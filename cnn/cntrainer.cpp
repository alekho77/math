#include "cntrainer.h"
#include "cntrainer_impl.h"

namespace cnn {

cntrainer::cntrainer(cnnetwork& network) : impl_(std::make_unique<cntrainer_impl>(*network.impl_)) {}

cntrainer::~cntrainer() = default;

void cntrainer::randomize(double range, unsigned seed) {
    impl_->randomize(range, seed);
}

std::tuple<double, double> cntrainer::operator()(const std::vector<double>& inputs,
                                                 const std::vector<double>& desired_outputs) {
    return impl_->exec(inputs, desired_outputs);
}

std::vector<double> cntrainer::states(size_t layer) const {
    return impl_->states(layer);
}

}  // namespace cnn
