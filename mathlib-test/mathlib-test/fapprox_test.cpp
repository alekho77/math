#include "math/mathlib/fapprox.h"

#include <gtest/gtest.h>

#include <cmath>

namespace mathlib {

class fapprox_test_fixture : public ::testing::Test {
 protected:
    static double foo_src1(double x) {
        return 8 * x - 3;
    }
    const matrix<double> M1 = {{8}, {-3}};
    static double foo_src2(double x) {
        return 10 * (1.0 - std::exp(-0.1 * x));
    }
    const matrix<double> M2 = {{10}, {0.1}};
};

TEST_F(fapprox_test_fixture, simple) {
    {
        fapprox<double(double, double)> appx;
        for (int i = -1; i <= 1; i++) {
            const double x = i;
            const double f = foo_src1(x);
            appx([x, f](double a, double b) { return a * x + b - f; });
        }
        auto x = appx.approach(0, 0).get_as_matrix();
        ASSERT_EQ(1, x.cols());
        ASSERT_EQ(2, x.rows());
        for (size_t i = 0; i < x.rows(); i++) {
            EXPECT_NEAR(M1[i][0], x[i][0], numeric_consts<double>::epsilon);
        }
    }
    {
        fapprox<double(double, double)> appx;
        for (int i = 0; i <= 14; i++) {
            const double x = i;
            const double f = foo_src2(x);
            appx([x, f](double a, double b) { return (a * (1.0 - std::exp(-b * x))) - f; });
        }
        auto x = appx.approach(9, 1).get_as_matrix();
        ASSERT_EQ(1, x.cols());
        ASSERT_EQ(2, x.rows());
        for (size_t i = 0; i < x.rows(); i++) {
            EXPECT_NEAR(M2[i][0], x[i][0], numeric_consts<double>::epsilon);
        }
    }
}

}  // namespace mathlib
