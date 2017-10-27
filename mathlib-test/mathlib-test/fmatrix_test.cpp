#include "math/mathlib/fmatrix.h"

#include <gtest/gtest.h>

namespace mathlib {

class fmatrix_test_fixture : public ::testing::Test {
protected:

};

TEST_F(fmatrix_test_fixture, contruct) {
  {
    fmatrix<int()> m{};
    EXPECT_EQ(0, m.rows());
    EXPECT_EQ(0, m.cols());
    EXPECT_TRUE(m.empty());
  }
}

}  // namespace mathlib
