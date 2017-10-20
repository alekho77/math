#include "math/mathlib/lsyseq.h"

#include <gtest/gtest.h>

namespace mathlib {

class lsyseq_test_fixture : public ::testing::Test {
protected:
  const matrix<double> A  = {{1, 1, 1},{5, 3, 2},{0, 1, -1}};
  const matrix<double> An = {{5, 3, 2},{0, 1, -1},{0, 0, 1}};
  const matrix<double> B  = {{25},{0},{6}};
  const matrix<double> Bn = {{0},{6},{113./5.}};

  const matrix<double> A1 = {{2,-1},{-1,2}};
  const matrix<double> B1 = {{0},{3}};
  const double     condA1 = 2.0/1.5;
  const matrix<double> X1 = {{1},{2}};

  const matrix<double> A2 = {{1,15},{5,75.01}};
  const matrix<double> B2 = {{17},{255}};
  const matrix<double> B2e = {{17},{255.03}};
  const double     condA2 = 2500.0;
  const matrix<double> X2 = {{17},{0}};
  const matrix<double> X2e = {{2},{3}};
};

TEST_F(lsyseq_test_fixture, intialization) {
  EXPECT_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2}}, {{25},{0}}), std::exception);
  EXPECT_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2},{5, 3, 2},{0, 1, -1}}, {{25},{0},{6},{0}}), std::exception);
  EXPECT_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2},{5, 3, 2}}, {{25, 0},{0, 6},{6, 0}}), std::exception);
  EXPECT_NO_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2},{0, 1, -1}}, {{25},{0},{6}}));
  EXPECT_NO_THROW(linear_equations<double>(A, B));
}

TEST_F(lsyseq_test_fixture, normalize) {
  {
    linear_equations<double> syseq{{{0,0},{0,0}}, {{0},{0}}};
    ASSERT_THROW(syseq.normalize(), std::exception);
  }
  {
    linear_equations<double> syseq{A, B};
    ASSERT_NO_THROW(syseq.normalize());
    for (size_t r = 0; r < An.rows(); r++) {
      for (size_t c = 0; c < An.cols(); c++) {
        EXPECT_DOUBLE_EQ(An[r][c], syseq.A()[r][c]);
      }
      EXPECT_DOUBLE_EQ(Bn[r][0], syseq.B()[r][0]);
    }
  }
}

TEST_F(lsyseq_test_fixture, conditionality) {
  {
    linear_equations<double> syseq = {A1, B1};
    EXPECT_DOUBLE_EQ(condA1, syseq.normalize().cond());
  }
  {
    linear_equations<double> syseq = {A2, B2};
    EXPECT_NEAR(condA2, syseq.normalize().cond(), 1e-6);
  }
}

TEST_F(lsyseq_test_fixture, solve) {
  {
    linear_equations<double> syseq = {A1, B1};
    ASSERT_NO_THROW(syseq.normalize().solve());
    EXPECT_EQ(X1, syseq.X());
  }
  {
    linear_equations<double> syseq = {A2, B2};
    ASSERT_NO_THROW(syseq.normalize().solve());
    ASSERT_EQ(X2.rows(), syseq.X().rows());
    ASSERT_EQ(X2.cols(), syseq.X().cols());
    for (size_t i = 0; i < X2.rows(); i++) {
      for (size_t j = 0; j < X2.cols(); j++) {
        EXPECT_DOUBLE_EQ(X2[i][j], syseq.X()[i][j]);
      }
    }
  }
}

}  // namespace mathlib
