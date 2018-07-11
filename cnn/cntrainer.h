/*
    Neural Network Trainer based on OpenCL.
    (c) 2018 Aleksey Khozin
*/

#ifndef CNN_CNTRAINER_H
#define CNN_CNTRAINER_H

#include "cnnetwork.h"

#include <random>
#include <memory>
#include <vector>
#include <tuple>

namespace cnn {

class cntrainer_impl;  // To hide OpenCL stuff

class cntrainer {
 public:
    explicit cntrainer(cnnetwork& network);
    ~cntrainer();

    cntrainer() = delete;
    cntrainer(const cntrainer&) = delete;
    cntrainer(cntrainer&&) = delete;

    void randomize(double range = 1, unsigned seed = std::random_device()());

    // Returns error before and after correction of weights
    std::tuple<double, double> operator()(const std::vector<double>& inputs,
                                          const std::vector<double>& desired_outputs);

    double learning_rate() const;
    void set_learning_rate(double eta);

    double momentum() const;
    void set_momentum(double alpha);

    std::vector<double> states(size_t layer) const;

 private:
    std::unique_ptr<cntrainer_impl> impl_;
};

}  // namespace cnn

#endif  // CNN_CNTRAINER_H
