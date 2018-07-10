/*
    Neural Network Trainer based on OpenCL.
    (c) 2018 Aleksey Khozin
*/

#ifndef CNN_CNTRAINER_H
#define CNN_CNTRAINER_H

#include "cnnetwork.h"

#include <random>

namespace cnn {

class cntrainer {
 public:
    explicit cntrainer(cnnetwork& network);
    void randomize(double range = 1, unsigned seed = std::random_device()());
};

}  // namespace cnn

#endif  // CNN_CNTRAINER_H
