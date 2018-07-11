/*
  Implementation of Neural Network based on OpenCL for artificial neural network modeling.
  (c) 2018 Aleksey Khozin
*/

#ifndef CNN_CNNETWORK_IMPL_H
#define CNN_CNNETWORK_IMPL_H

#include "cnnetwork.h"

#include <CL/cl.hpp>

#include <memory>
#include <vector>

namespace cnn {

class cnnetwork_impl {
 public:
    cnnetwork_impl(size_t inputs, const std::vector<cnlayer>& layers);

    size_t inputs_num() const {
        return inputs_;
    }
    size_t layer_num() const {
        return layers_.size();
    }
    const cnlayer& get_layer(size_t idx) const {
        return layers_[idx].desc;
    }

    std::vector<cl_double> weights(size_t layer, size_t neuron) const;
    void set_weights(size_t layer, size_t neuron, const std::vector<cl_double>& weights);

    std::vector<cl_double> exec(const std::vector<cl_double>& inputs);

 private:
    size_t weights_number(size_t l) const;
    size_t make_input_layer(size_t inputs, bool bias);
    size_t make_layer(size_t input_size, cnlayer layer, bool bias);

    struct cllayer {
        cnlayer desc;
        cl_int stride;             // size of input buffer
        cl::Buffer inputs;         // reference to exist buffer
        cl::Buffer weights;        // weights of all nodes
        cl::Buffer inter_outputs;  // the same size like weights buffer
        cl::Buffer outputs;        // result of all nodes plus 1.0 for bias of next layer if it is present
    };

    const size_t inputs_;

    cl::Context context_ = cl::Context(CL_DEVICE_TYPE_GPU);
    cl::CommandQueue cmd_queue_;
    cl::Program prog_;
    cl::Kernel kernel_;
    cl::Buffer input_buf_;
    std::vector<cllayer> layers_;
};

}  // namespace cnn

#endif  // CNN_CNNETWORK_IMPL_H
