#include "math/mathlib/diff.h"

#include <gtest/gtest.h>

namespace mathlib {

class diff_test_fixture : public ::testing::Test {
public:
  const double x1 = 1234.0;

  static double foo1(double x) {
    return x;
  }
  double foo2(double x) {
    return x;
  }
};

TEST_F(diff_test_fixture, construct) {
  auto d1 = make_diff(&foo1);
  auto d2 = make_diff(&diff_test_fixture::foo2, static_cast<diff_test_fixture*>(this));
  d1(x1);
  d2(x1);
}

}  // namespace mathlib
