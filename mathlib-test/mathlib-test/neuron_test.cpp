#include "math/mathlib/neuron.h"

#include <gtest/gtest.h>

namespace mathlib {

TEST(neuron_test, construct) {
  {
    neuron<double, 1> n1;
    EXPECT_EQ(0, n1(0));
    n1.set_bias(1);
    EXPECT_EQ(0, n1(-1));
    EXPECT_EQ(1, n1.slope());
    n1.set_weights(-1);
    EXPECT_EQ(0, n1(1));
  }
  {
    neuron<float, 10, NOBIAS<float>, SIGMOID<float,SLOPE<float>>> n2;
    EXPECT_EQ(0, n2(1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f));
    EXPECT_EQ(0, n2.bias());
    n2.set_slope(0);
    EXPECT_EQ(std::make_tuple(1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f), n2.weights());
    EXPECT_EQ(0, n2(1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f));
    n2.set_slope(1);
    n2.set_weights(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    EXPECT_EQ(0, n2(1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f));
  }
}

TEST(neuron_test, deriv) {
  SIGMOID<double, SLOPE<double>> sig;
  sig.slope_ = 3.14;
  const double val = sig(2.5);
  auto deriv = [](double x) { return 2 * 3.14 * std::exp(-3.14 * x) / std::pow(1 + std::exp(-3.14 * x), 2.0); };
  EXPECT_NEAR(deriv(2.5), sig.deriv(val), 1e-14);
}

}  // namespace mathlib
