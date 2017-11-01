#include "math/mathlib/fapprox.h"

#include <gtest/gtest.h>

#include <cmath>

namespace mathlib {

class fapprox_test_fixture : public ::testing::Test {
protected:
  static double foo_src1(double x) { return 10 * (1.0 - std::exp(-0.1 * x)); }
};

TEST_F(fapprox_test_fixture, simple) {
  fapprox<double(double, double)> appx;
  for (int i = 0; i <= 100; i+=5) {
    const double x = i;
    const double f = foo_src1(x);
    appx([x, f](double a, double b) { return a * (1 - std::exp(- b * x)) - f; });
  }
  appx.approach(9.0, 0.0);
}

}  // namespace mathlib
