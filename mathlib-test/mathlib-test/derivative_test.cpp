#define _USE_MATH_DEFINES

#include "math/mathlib/derivative.h"

#include <gtest/gtest.h>

#include <cmath>

namespace mathlib {

class diff_test_fixture : public ::testing::Test {
 public:
    static double foo1(double x) {
        return x;
    }
    const double dfoo1 = 1.0;

    double foo2(double x, double y, double z) {
        return x + y + z;
    }
    const double dfoo2 = 1.0;

    static double foo3(double x) {
        return x * x;
    }
    const double dfoo3_0 = 0.0;
    const double dfoo3_1 = 20;  // x = 10
    const double dfoo3_2 = -20; // x = -10

    static double foo4(double x, double y, double z) {
        return 2 * x * x * x - 3 * y * y * y + 4 * z * z * z - 5 * x * y * z;
    }
    const double dfoo4_x = 11; // 1, -1, 1
    const double dfoo4_y = -14;
    const double dfoo4_z = 17;

    static double foo5(double x) {
        return std::exp(-x);
    }
    const double dfoo5_0 = -1;         // x = 0
    const double dfoo5_1 = -1.0 / M_E; // x = 1
    const double dfoo5_2 = -M_E;       // x = -1
};

TEST_F(diff_test_fixture, construct) {
    const auto d1 = make_deriv(&foo1);
    auto d2 = make_deriv(&diff_test_fixture::foo2, static_cast<diff_test_fixture*>(this));
    auto d3 = d1;
}

TEST_F(diff_test_fixture, simple) {
    {
        auto d = make_deriv(&foo1);
        EXPECT_NEAR(dfoo1, d.diff<0>(1234), numeric_consts<double>::epsilon);
    }
    {
        auto d = make_deriv(&diff_test_fixture::foo2, static_cast<diff_test_fixture*>(this));
        EXPECT_NEAR(dfoo2, d.diff<0>(1234, 2, -3), numeric_consts<double>::epsilon);
        EXPECT_NEAR(dfoo2, d.diff<1>(1234, 2, -3), 1e-9);
        EXPECT_NEAR(dfoo2, d.diff<2>(1234, 2, -3), 1e-9);
    }
    {
        auto d = make_deriv(&foo3);
        EXPECT_NEAR(dfoo3_0, d.diff<0>(0), numeric_consts<double>::epsilon);
        EXPECT_NEAR(dfoo3_1, d.diff<0>(10), 1e-11);
        EXPECT_NEAR(dfoo3_2, d.diff<0>(-10), 1e-11);
    }
    {
        auto d = make_deriv(&foo4);
        EXPECT_NEAR(dfoo4_x, d.diff<0>(1, -1, 1), 1e-9);
        EXPECT_NEAR(dfoo4_y, d.diff<1>(1, -1, 1), 1e-9);
        EXPECT_NEAR(dfoo4_z, d.diff<2>(1, -1, 1), 1e-9);
    }
    {
        auto d = make_deriv(&foo5);
        EXPECT_NEAR(dfoo5_0, d.diff<0>(0), 1e-10);
        EXPECT_NEAR(dfoo5_1, d.diff<0>(1), 1e-11);
        EXPECT_NEAR(dfoo5_2, d.diff<0>(-1), 1e-10);
    }
}

} // namespace mathlib
