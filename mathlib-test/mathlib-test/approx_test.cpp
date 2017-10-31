#include "math/mathlib/approx.h"

#include <gtest/gtest.h>

namespace mathlib {

class approx_test_fixture : public ::testing::Test {
protected:
  const matrix<double> M1 = {{8},{-3}};
  double foo2(double x) { return 3 * x*x - 4 * x + 5; }
  const matrix<double> M2 = {{3},{-4},{5}};
};

TEST_F(approx_test_fixture, simple) {
  {
    approx<double, 2> appx;
    appx(0.0, 1.0, -3.0)  // 8 * x - 3
        (1.0, 1.0, 5.0)
        (-1.0, 1.0, -11.0);
    EXPECT_EQ(M1, appx.approach());
  }
  {
    approx<double, 3> appx;
    for (int i = 0; i < 11; i++) {
      const double x = (double)(i - 5);
      appx(x * x, x, 1.0, foo2(x));
    }
    auto m = appx.approach();
    ASSERT_EQ(M2.rows(), m.rows());
    ASSERT_EQ(M2.cols(), m.cols());
    for (size_t i = 0; i < M2.rows(); i++) {
      EXPECT_DOUBLE_EQ(M2[i][0], m[i][0]);
    }
  }
}

}  // namespace mathlib
