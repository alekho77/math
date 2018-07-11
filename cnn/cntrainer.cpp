#include "cntrainer.h"
#include "cntrainer_impl.h"

namespace cnn {

cntrainer::cntrainer(cnnetwork& network) : impl_(std::make_unique<cntrainer_impl>(*network.impl_)) {}

cntrainer::~cntrainer() = default;

void cntrainer::randomize(double /*range*/, unsigned /*seed*/) {}

}  // namespace cnn
