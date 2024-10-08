#include "mathlib/approx.h"

#include <gtest/gtest.h>

namespace mathlib {

class approx_test_fixture : public ::testing::Test {
 protected:
    const std::tuple<double, double> M1 = {8, -3};
    double foo2(double x) {
        return 3 * x * x - 4 * x + 5;
    }
    const matrix<double> M2 = {{3}, {-4}, {5}};
};

TEST_F(approx_test_fixture, simple) {
    {
        approx<double, 2> appx;
        appx(0u, 1.0f, -3) // 8 * x - 3
            (1, 1, 5)(-1, 1, -11);
        EXPECT_EQ(M1, appx.approach().get_as_tuple());
    }
    {
        approx<double, 3> appx;
        for (int i = 0; i < 11; i++) {
            const int x = i - 5;
            appx(x * x, x, 1, foo2(x));
        }
        auto m = appx.approach().get_as_matrix();
        ASSERT_EQ(M2.rows(), m.rows());
        ASSERT_EQ(M2.cols(), m.cols());
        for (size_t i = 0; i < M2.rows(); i++) {
            EXPECT_DOUBLE_EQ(M2[i][0], m[i][0]);
        }
    }
}

} // namespace mathlib
