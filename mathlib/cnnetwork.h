/*
  Neural Network based on OpenCL for artificial neural network modeling.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_CNNETWORK_H
#define MATHLIB_CNNETWORK_H

#include <memory>

namespace mathlib {

enum class cnfunction {
  sigmoid
};

struct cnneuron {
  cnfunction type;
  size_t synapses;
  bool bias;
};

class cnnetwork {
public:
  cnnetwork() = delete;
  cnnetwork(size_t inputs);
  cnnetwork(cnnetwork&&) = default;
  ~cnnetwork();

private:

  class impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace mathlib

#endif  // MATHLIB_CNNETWORK_H
