/*
  Neural Network based on OpenCL for artificial neural network modeling.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_CNNETWORK_H
#define MATHLIB_CNNETWORK_H

#include <memory>
#include <vector>

namespace mathlib {

enum class cnfunction { sigmoid = 1 };

struct cnneuron {
    cnfunction type;
    size_t synapses;
    bool bias;
    inline bool operator==(const cnneuron& that) const {
        return this->type == that.type && this->synapses == that.synapses && this->bias == that.bias;
    }
};

struct cnnode {
    cnneuron neuron;
    std::vector<int> map;
    inline bool operator==(const cnnode& that) const {
        return this->neuron == that.neuron && this->map == that.map;
    }
};

using cnlayer = std::vector<cnnode>;

class cnnetwork {
 public:
    cnnetwork() = delete;
    cnnetwork(size_t inputs, std::initializer_list<cnlayer>&& layers);
    cnnetwork(cnnetwork&&) = default;
    ~cnnetwork();

    size_t inputs_num() const;
    size_t layer_num() const;
    cnlayer layer_desc(size_t idx) const;

    std::vector<double> weights(size_t layer, size_t neuron) const;
    void set_weights(size_t layer, size_t neuron, std::vector<double> weights);
    double bias(size_t layer, size_t neuron) const;
    void set_bias(size_t layer, size_t neuron, double bias) const;

    std::vector<double> operator()(const std::vector<double>& inputs);

    template <typename... Args>
    std::vector<double> operator()(Args... args) {
        return operator()({static_cast<double>(args)...});
    }

 private:
    class impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace mathlib

#endif  // MATHLIB_CNNETWORK_H
