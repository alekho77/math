/*
    Neural Network Trainer based on OpenCL.
    (c) 2018 Aleksey Khozin
*/

#ifndef MATHLIB_CNTRAINER_H
#define MATHLIB_CNTRAINER_H

#include "cnnetwork.h"

#include <random>

namespace mathlib {

class cntrainer {
 public:
    explicit cntrainer(cnnetwork& network);
    void randomize(double range = 1, unsigned seed = std::random_device()());
};

}  // namespace mathlib

#endif  // MATHLIB_CNTRAINER_H
