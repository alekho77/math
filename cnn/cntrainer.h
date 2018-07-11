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

namespace cnn {

class cntrainer {
 public:
    explicit cntrainer(cnnetwork& network);
    ~cntrainer();

    cntrainer() = delete;
    cntrainer(const cntrainer&) = delete;
    cntrainer(cntrainer&&) = delete;

    void randomize(double range = 1, unsigned seed = std::random_device()());

 private:
    class impl;
    std::unique_ptr<impl> impl_;

    cnnetwork& network_;
};

}  // namespace cnn

#endif  // CNN_CNTRAINER_H
