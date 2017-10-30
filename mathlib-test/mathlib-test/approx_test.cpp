#include "math/mathlib/approx.h"

#include <gtest/gtest.h>

namespace mathlib {

class approx_test_fixture : public ::testing::Test {
protected:
};

TEST_F(approx_test_fixture, contruct) {
  approx<double> appx;
  appx(10.0, 10.0, 20.0);
}

}  // namespace mathlib
