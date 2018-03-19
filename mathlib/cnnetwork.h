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
  sigmoid = 1
};

struct cnneuron {
  cnfunction type;
  int synapses;
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
  ~cnnetwork();

  cnlayer layer(size_t idx) const;

  std::vector<double> operator ()(const std::vector<double>& inputs);

  template <typename... Args>
  std::vector<double> operator ()(Args... args) {
    return operator() ({static_cast<double>(args)...});
  }

private:
  class impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace mathlib

#endif  // MATHLIB_CNNETWORK_H
