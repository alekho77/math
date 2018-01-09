/*
  Neural Network based on OpenCL for artificial neural network modeling.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_CNNETWORK_H
#define MATHLIB_CNNETWORK_H

#include <memory>
#include <vector>

namespace mathlib {

enum class cnfunction {
  sigmoid
};

struct cnneuron {
  cnfunction type;
  size_t synapses;
  bool bias;
};

struct cnnode {
  cnneuron neuron;
  std::vector<int> map;
};

using cnlayer = std::vector<cnnode>;

class cnnetwork {
public:
  cnnetwork() = delete;
  cnnetwork(size_t inputs, std::initializer_list<cnlayer>&& layers);
  cnnetwork(cnnetwork&&) = default;
  ~cnnetwork();

private:

  class impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace mathlib

#endif  // MATHLIB_CNNETWORK_H
