#include "cntrainer.h"

namespace cnn {

class cntrainer::impl {
 public:
 private:
};

cntrainer::cntrainer(cnnetwork& network) : network_(network) {}

cntrainer::~cntrainer() = default;

void cntrainer::randomize(double /*range*/, unsigned /*seed*/) {}

}  // namespace cnn
