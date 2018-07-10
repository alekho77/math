/*
  Neural Network based on OpenCL for artificial neural network modeling.
  (c) 2017 Aleksey Khozin
*/

#ifndef CNN_CNNETWORK_H
#define CNN_CNNETWORK_H

#include <memory>
#include <vector>

namespace cnn {

enum class cnfunction { sigmoid = 1 };

struct cnlayer {
    size_t nodes;
    cnfunction type;
    bool bias;
};

class cnnetwork {
 public:
    cnnetwork() = delete;
    cnnetwork(size_t inputs, const std::vector<cnlayer>& layers);
    cnnetwork(const cnnetwork&) = delete;
    cnnetwork(cnnetwork&&) = delete;
    ~cnnetwork();

    size_t inputs_num() const;
    size_t layer_num() const;
    cnlayer layer_desc(size_t idx) const;

    // The last one is weight to bias neuron if it is present
    std::vector<double> weights(size_t layer, size_t neuron) const;
    void set_weights(size_t layer, size_t neuron, const std::vector<double>& weights);

    std::vector<double> operator()(const std::vector<double>& inputs);

    template <typename... Args>
    std::vector<double> operator()(Args... args) {
        return operator()({static_cast<double>(args)...});
    }

 private:
    class impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace cnn

#endif  // CNN_CNNETWORK_H
