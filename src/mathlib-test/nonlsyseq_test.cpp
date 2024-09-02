#define _USE_MATH_DEFINES

#include "nonlsyseq.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iterator>

namespace mathlib {
class nonlsyseq_test_fixture : public ::testing::Test {
 protected:
    typedef double (*F1_t)(double, double);
    const F1_t F1[2] = {[](double x, double y) { return 2 * x - 1 * y; },
                        [](double x, double y) { return -1 * x + 2 * y - 3; }};
    const matrix<double> X1 = {{1}, {2}};

    static double foo1(double x) {
        return (x - 10) * (x + 10);
    }
    const double foo1_a[3] = {-1, +1, 0};
    const double foo1_x[3] = {-10, 10, 10};
    static double foo2(double x) {
        return std::exp(-std::abs((x - 2) * (x + 2))) - 1.0 / M_E;
    }
    const double foo2_a[4] = {-5, -1.9, 1.9, 5};
    const double foo2_x[4] = {-std::sqrt(5), -std::sqrt(3), std::sqrt(3), std::sqrt(5)};
};

TEST_F(nonlsyseq_test_fixture, intialization) {
    EXPECT_THROW(nonlinear_equations<float(float)>({[](float x) { return x; }, [](float x) { return 2 * x; }}),
                 std::exception);
    EXPECT_NO_THROW(nonlinear_equations<double(double, double, double)>(
        {[](double x, double y, double z) { return x * y * z; }, [](double x, double y, double z) { return x * y * z; },
         [](double x, double y, double z) { return x * y * z; }}));
    EXPECT_NO_THROW(nonlinear_equations<double(double)>{[](double x) { return x; }});
}

TEST_F(nonlsyseq_test_fixture, simple) {
    {
        nonlinear_equations<double(double, double)> syseq{F1[0], F1[1]};
        matrix<double> x = syseq.solve(0, 0);
        ASSERT_EQ(1, x.cols());
        ASSERT_EQ(2, x.rows());
        for (size_t i = 0; i < x.rows(); i++) {
            EXPECT_NEAR(X1[i][0], x[i][0], numeric_consts<double>::epsilon);
        }
    }
    {
        nonlinear_equations<double(double)> syseq{&foo1};
        for (size_t i = 0; i < std::size(foo1_a); i++) {
            const auto x = syseq.solve(foo1_a[i]);
            EXPECT_DOUBLE_EQ(foo1_x[i], x[0][0]);
        }
    }
}

TEST_F(nonlsyseq_test_fixture, complex) {
    {
        nonlinear_equations<double(double)> syseq{&foo2};
        for (size_t i = 0; i < std::size(foo2_a); i++) {
            const auto x = syseq.solve(foo2_a[i]);
            EXPECT_NEAR(foo2_x[i], x[0][0], numeric_consts<double>::epsilon);
        }
    }
}

} // namespace mathlib
