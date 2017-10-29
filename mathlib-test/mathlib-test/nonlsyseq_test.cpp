#include "math/mathlib/nonlsyseq.h"

#include <gtest/gtest.h>

namespace mathlib {
class nonlsyseq_test_fixture : public ::testing::Test {
protected:
};

TEST_F(nonlsyseq_test_fixture, intialization) {
  EXPECT_THROW(nonlinear_equations<float(float)>({[](float x) { return x; }, [](float x) { return 2*x; }}), std::exception);
  
  //EXPECT_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2}}, {{25},{0}}), std::exception);
  //EXPECT_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2},{5, 3, 2},{0, 1, -1}}, {{25},{0},{6},{0}}), std::exception);
  //EXPECT_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2},{5, 3, 2}}, {{25, 0},{0, 6},{6, 0}}), std::exception);
  //EXPECT_NO_THROW(linear_equations<double>({{1, 1, 1},{5, 3, 2},{0, 1, -1}}, {{25},{0},{6}}));
  //EXPECT_NO_THROW(linear_equations<double>(A, B));
}

}  // namespace mathlib
