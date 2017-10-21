#include "math/mathlib/matrix.h"

#include <gtest/gtest.h>

namespace mathlib {

class matrix_test_fixture : public ::testing::Test {
protected:
  const matrix<int> M1 = {{1, 2, 3},{4, 5, 6}};
  const matrix<int> M1S = {{14, 32},{32, 77}}; // M1*trans(M1)

};

TEST_F(matrix_test_fixture, contruct) {
  {
    matrix<int> m;
    EXPECT_EQ(0, m.rows());
    EXPECT_EQ(0, m.cols());
    EXPECT_TRUE(m.empty());
  }
  {
    const matrix<int> m{2};
    EXPECT_EQ(1, m.cols());
    EXPECT_EQ(2, m.rows());
    EXPECT_FALSE(m.empty());
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
    matrix<int> m3 = std::move(m2);
    ASSERT_EQ(3, m3.cols());
    ASSERT_EQ(2, m3.rows());
    for (size_t r = 0; r < m3.rows(); r++) {
      for (size_t c = 0; c < m3.cols(); c++) {
        EXPECT_EQ(1234, m3[r][c]);
      }
    }
  }
}

TEST_F(matrix_test_fixture, assign) {
  {
    matrix<int> m{2,3};
    for (size_t i = 0; i < m.rows(); i++) {
      for (size_t j = 0; j < m.cols(); j++) {
        m[i][j] = static_cast<int>(i + 1) * static_cast<int>(j + 1);
      }
    }
    for (size_t i = 0; i < m.rows(); i++) {
      for (size_t j = 0; j < m.cols(); j++) {
        EXPECT_EQ(static_cast<int>(i + 1) * static_cast<int>(j + 1), m[i][j]);
      }
    }
  }
  {
    matrix<int> m = {{1, 2, 3},
                     {4, 5, 6}};
    ASSERT_EQ(3, m.cols());
    ASSERT_EQ(2, m.rows());
    int v = 1;
    for (size_t r = 0; r < m.rows(); r++) {
      for (size_t c = 0; c < m.cols(); c++) {
        EXPECT_EQ(v, m[r][c]);
        v++;
      }
    }
  }
  {
    auto make_matrix = []() {
      matrix<int> m = {{1, 2, 3}, {4, 5}};
      return m; };
    ASSERT_THROW(make_matrix(), std::exception);
  }
}

TEST_F(matrix_test_fixture, copy) {
  {
    const matrix<int> m1{2, 3, 1234};
    matrix<int> m2;
    m2 = m1;
    ASSERT_EQ(2, m2.rows());
    ASSERT_EQ(3, m2.cols());
    for (size_t r = 0; r < m2.rows(); r++) {
      for (size_t c = 0; c < m2.cols(); c++) {
        EXPECT_EQ(1234, m2[r][c]);
      }
    }
  }
}

TEST_F(matrix_test_fixture, move) {
  {
    const matrix<int> m1 = transpose(M1);
    matrix<int> m2;
    m2 = M1 * m1;
    ASSERT_EQ(M1.rows(), m2.rows());
    ASSERT_EQ(M1.rows(), m2.cols());
    EXPECT_EQ(M1S, m2);
  }
}

TEST_F(matrix_test_fixture, transpose) {
  matrix<int> m1 = {{1, 2, 3},
                    {4, 5, 6}};
  matrix<int> m2 = transpose(m1);
  ASSERT_EQ(2, m2.cols());
  ASSERT_EQ(3, m2.rows());
  int v = 1;
  for (size_t c = 0; c < m2.cols(); c++) {
    for (size_t r = 0; r < m2.rows(); r++) {
      EXPECT_EQ(v, m2[r][c]);
      v++;
    }
  }
}

TEST_F(matrix_test_fixture, multiplication) {
  {
    ASSERT_THROW(M1 * M1, std::exception);
  }
  {
    matrix<int> m1 = transpose(M1);
    ASSERT_NO_THROW(M1 * m1);
    auto m2 = M1 * m1;
    ASSERT_EQ(M1.rows(), m2.rows());
    ASSERT_EQ(M1.rows(), m2.cols());
    EXPECT_EQ(M1S, m2);
  }
}

TEST_F(matrix_test_fixture, swap) {
  matrix<int> m1 = {{1, 2, 3},
                    {4, 5, 6}};
  const matrix<int> m2 = {{4, 5, 6},
                          {1, 2, 3}};
  m1.swap_row(0, 1);
  EXPECT_EQ(m2, m1);
}

TEST_F(matrix_test_fixture, addition) {
  {
    matrix<int> m1{2,3};
    matrix<int> m2{1,2};
    ASSERT_THROW(m1 + m2, std::exception);
  }
  {
    matrix<int> m1 = M1 + M1;
    for (size_t i = 0; i < M1.rows(); i++) {
      for (size_t j = 0; j < M1.cols(); j++) {
        EXPECT_EQ(M1[i][j] * 2, m1[i][j]);
      }
    }
  }
}

TEST_F(matrix_test_fixture, subtraction) {
  {
    matrix<int> m1{2,3};
    matrix<int> m2{1,2};
    ASSERT_THROW(m1 - m2, std::exception);
  }
  {
    matrix<int> m1 = M1 - M1;
    for (size_t i = 0; i < M1.rows(); i++) {
      for (size_t j = 0; j < M1.cols(); j++) {
        EXPECT_EQ(0, m1[i][j]);
      }
    }
  }
}

}  // namespace mathlib