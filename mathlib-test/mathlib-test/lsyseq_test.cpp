#include "math/mathlib/lsyseq.h"

#include <gtest/gtest.h>

namespace mathlib {

class lsyseq_test_fixture : public ::testing::Test {
protected:
  const matrix<double> A  = {{1, 1, 1},{5, 3, 2},{0, 1, -1}};
  const matrix<double> An = {{5, 3, 2},{0, 1, -1},{0, 0, 1}};
  const matrix<double> B  = {{25},{0},{6}};
  const matrix<double> Bn = {{0},{6},{113./5.}};
};

TEST_F(lsyseq_test_fixture, intialization) {
  EXPECT_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2}}, {{25},{0}}), std::exception);
  EXPECT_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2},{5, 3, 2},{0, 1, -1}}, {{25},{0},{6},{0}}), std::exception);
  EXPECT_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2},{5, 3, 2}}, {{25, 0},{0, 6},{6, 0}}), std::exception);
  EXPECT_NO_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2},{0, 1, -1}}, {{25},{0},{6}}));
  EXPECT_NO_THROW(linear_equations<double>(A, B));
}

TEST_F(lsyseq_test_fixture, normalize) {
  linear_equations<double> syseq{A, B};
  ASSERT_NO_THROW(syseq.normalize());
  for (size_t r = 0; r < An.rows(); r++) {
    for (size_t c = 0; c < An.cols(); c++) {
      EXPECT_DOUBLE_EQ(An[r][c], syseq.A()[r][c]);
    }
    EXPECT_DOUBLE_EQ(Bn[r][0], syseq.B()[r][0]);
  }
}

}  // namespace mathlib
