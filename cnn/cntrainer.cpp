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

double cntrainer::learning_rate() const {
    return impl_->learning_rate();
}

void cntrainer::set_learning_rate(double eta) {
    impl_->set_learning_rate(eta);
}

double cntrainer::momentum() const {
    return impl_->momentum();
}

void cntrainer::set_momentum(double alpha) {
    impl_->set_momentum(alpha);
}

std::vector<double> cntrainer::states(size_t layer) const {
    return impl_->states(layer);
}

std::vector<double> cntrainer::deltas(size_t layer) const {
    return impl_->deltas(layer);
}

}  // namespace cnn
