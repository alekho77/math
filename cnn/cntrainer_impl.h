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

    // Returns error before and after correction of weights
    std::tuple<cl_double, cl_double> operator()(const std::vector<cl_double>& inputs,
                                                const std::vector<cl_double>& desired_outputs);

    double learning_rate() const;
    void set_learning_rate(cl_double eta);

    double momentum() const;
    void set_momentum(cl_double alpha);

 private:
    cnnetwork_impl& network_;
};

}  // namespace cnn

#endif  // CNN_CNTRAINER_IMPL_H
