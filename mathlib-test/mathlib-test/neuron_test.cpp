#include "math/mathlib/neuron.h"

#include <gtest/gtest.h>

namespace mathlib {

class neuron_test_fixture : public ::testing::Test {
protected:
};

TEST_F(neuron_test_fixture, construct) {
  neuron<double, 1> n1;
}

}  // namespace mathlib
