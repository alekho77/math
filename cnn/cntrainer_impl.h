/*
    Implementation of Neural Network Trainer based on OpenCL.
    (c) 2018 Aleksey Khozin
*/

#ifndef CNN_CNTRAINER_IMPL_H
#define CNN_CNTRAINER_IMPL_H

#include "cnnetwork_impl.h"

#include <vector>
#include <tuple>

namespace cnn {

class cntrainer_impl final {
 public:
    explicit cntrainer_impl(cnnetwork_impl& network);

    cntrainer_impl() = delete;
    cntrainer_impl(const cntrainer_impl&) = delete;
    cntrainer_impl(cntrainer_impl&&) = delete;

    void randomize(cl_double range, unsigned seed);
    void reset();

    // Returns error before and after correction of weights
    std::tuple<cl_double, cl_double> exec(const std::vector<cl_double>& inputs,
                                          const std::vector<cl_double>& desired_outputs);

    cl_double learning_rate() const;
    void set_learning_rate(cl_double eta);

    cl_double momentum() const;
    void set_momentum(cl_double alpha);

    std::vector<cl_double> states(size_t layer) const;
    std::vector<cl_double> deltas(size_t layer) const;

 private:
    void compute_deltas(const std::vector<cl_double>& output_deltas);
    void adjust_weights();

    struct train_layer {
        const cnnetwork_impl::cllayer& network_layer;
        cl::Buffer deltas_inter;  // intermediate buffer to compute deltas
        cl::Buffer deltas;        // buffer with deltas
        cl::Buffer adjustments;   // buffer with previous adjustments
    };

    cnnetwork_impl& network_;
    std::vector<train_layer> layers_;
    cl::Kernel delta_kernel_;
    cl::Kernel adjust_kernel_;

    cl_double eta_ = 1.0;
    cl_double alpha_ = 0.0;
};

}  // namespace cnn

#endif  // CNN_CNTRAINER_IMPL_H
