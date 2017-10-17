#include "math/mathlib/matrix.h"

#include <gtest/gtest.h>

namespace mathlib {

class matrix_test_fixture : public ::testing::Test {
protected:

};

TEST_F(matrix_test_fixture, contruct) {
  {
    const matrix<int> m{2};
    EXPECT_EQ(1, m.cols());
    EXPECT_EQ(2, m.rows());
  }
  {
    const matrix<int> m{2, 3};
    for (size_t r = 0; r < m.rows(); r++) {
      for (size_t c = 0; c < m.cols(); c++) {
        EXPECT_EQ(int(), m[r][c]);
      }
    }
  }
  {
    const matrix<int> m1{2, 3, 1234};
    for (size_t r = 0; r < m1.rows(); r++) {
      for (size_t c = 0; c < m1.cols(); c++) {
        EXPECT_EQ(1234, m1[r][c]);
      }
    }
    matrix<int> m2 = m1;
    ASSERT_EQ(3, m2.cols());
    ASSERT_EQ(2, m2.rows());
    for (size_t r = 0; r < m2.rows(); r++) {
      for (size_t c = 0; c < m2.cols(); c++) {
        EXPECT_EQ(1234, m2[r][c]);
      }
    }
  }
}

}  // namespace mathlib