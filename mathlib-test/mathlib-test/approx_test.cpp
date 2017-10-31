#include "math/mathlib/approx.h"

#include <gtest/gtest.h>

namespace mathlib {

class approx_test_fixture : public ::testing::Test {
protected:
};

TEST_F(approx_test_fixture, contruct) {
  approx<double, 2> appx;
   appx(1.0, 2.0, 3.0)
       (4.0, 5.0, 6.0);

   appx.approach();
}

}  // namespace mathlib
