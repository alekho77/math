#include "math/mathlib/diff.h"

#include <gtest/gtest.h>

namespace mathlib {

class diff_test_fixture : public ::testing::Test {
public:
  const double x1 = 1234.0;
  const double y1 = 2.0;
  const double z1 = 3.0;

  static double foo1(double x) {
    return x;
  }
  const double dfoo1_x = 1.0;

  double foo2(double x, double y, double z) {
    return x + y + z;
  }
  const double dfoo2_y = 1.0;
};

TEST_F(diff_test_fixture, construct) {
  auto d1 = make_diff(&foo1);
  auto d2 = make_diff(&diff_test_fixture::foo2, static_cast<diff_test_fixture*>(this));
  EXPECT_NEAR(dfoo1_x, d1.deriv<0>(x1), 1e-6);
  EXPECT_NEAR(dfoo2_y, d2.deriv<1>(x1, y1, z1), 1e-6);
}

}  // namespace mathlib
