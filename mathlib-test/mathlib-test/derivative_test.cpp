#include "math/mathlib/derivative.h"

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
  const auto d1 = make_deriv(&foo1);
  auto d2 = make_deriv(&diff_test_fixture::foo2, static_cast<diff_test_fixture*>(this));
  auto d3 = d1;
}

TEST_F(diff_test_fixture, simple) {
  auto d1 = make_deriv(&foo1);
  auto d2 = make_deriv(&diff_test_fixture::foo2, static_cast<diff_test_fixture*>(this));
  EXPECT_NEAR(dfoo1_x, d1.diff<0>(x1), numeric_helper<double>::epsilon);
  EXPECT_NEAR(dfoo2_y, d2.diff<1>(x1, y1, z1), 1e-9);
}

}  // namespace mathlib
