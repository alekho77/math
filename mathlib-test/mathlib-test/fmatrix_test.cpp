#include "math/mathlib/fmatrix.h"

#include <gtest/gtest.h>

namespace mathlib {

class fmatrix_test_fixture : public ::testing::Test {
 protected:
    const fmatrix<int()> M1 = {{[]() { return 1; }, []() { return 2; }, []() { return 3; }},
                               {[]() { return 4; }, []() { return 5; }, []() { return 6; }}};
    const fmatrix<int()> M1S = {{[]() { return 14; }, []() { return 32; }},
                                {[]() { return 32; }, []() { return 77; }}};  // M1*trans(M1)
};

TEST_F(fmatrix_test_fixture, contruct) {
    {
        fmatrix<int()> m{};
        EXPECT_EQ(0, m.rows());
        EXPECT_EQ(0, m.cols());
        EXPECT_TRUE(m.empty());
    }
    {
        const fmatrix<int()> m{2};
        EXPECT_EQ(1, m.cols());
        EXPECT_EQ(2, m.rows());
        EXPECT_FALSE(m.empty());
    }
    {
        const fmatrix<int()> m{2, 3};
        for (size_t r = 0; r < m.rows(); r++) {
            for (size_t c = 0; c < m.cols(); c++) {
                EXPECT_TRUE(m[r][c] == nullptr);
            }
        }
    }
    {
        const fmatrix<int()> m1{2, 3};
        auto m2 = m1;
        ASSERT_EQ(3, m2.cols());
        ASSERT_EQ(2, m2.rows());
        for (size_t r = 0; r < m2.rows(); r++) {
            for (size_t c = 0; c < m2.cols(); c++) {
                EXPECT_TRUE(m2[r][c] == nullptr);
            }
        }
        auto m3 = std::move(m2);
        ASSERT_EQ(3, m3.cols());
        ASSERT_EQ(2, m3.rows());
        for (size_t r = 0; r < m3.rows(); r++) {
            for (size_t c = 0; c < m3.cols(); c++) {
                EXPECT_TRUE(m3[r][c] == nullptr);
            }
        }
    }
}

TEST_F(fmatrix_test_fixture, assign) {
    {
        fmatrix<int()> m{2, 3};
        for (size_t i = 0; i < m.rows(); i++) {
            for (size_t j = 0; j < m.cols(); j++) {
                m[i][j] = [=]() -> int { return static_cast<int>(i + 1) * static_cast<int>(j + 1); };
            }
        }
        for (size_t i = 0; i < m.rows(); i++) {
            for (size_t j = 0; j < m.cols(); j++) {
                EXPECT_EQ(static_cast<int>(i + 1) * static_cast<int>(j + 1), m[i][j]());
            }
        }
    }
    {
        fmatrix<int()> m = {{[]() { return 1; }, []() { return 2; }, []() { return 3; }},
                            {[]() { return 4; }, []() { return 5; }, []() { return 6; }}};
        ASSERT_EQ(3, m.cols());
        ASSERT_EQ(2, m.rows());
        int v = 1;
        for (size_t r = 0; r < m.rows(); r++) {
            for (size_t c = 0; c < m.cols(); c++) {
                EXPECT_EQ(v, m[r][c]());
                v++;
            }
        }
    }
    {
        auto make_fmatrix = []() {
            fmatrix<int()> m = {{[]() { return 1; }, []() { return 2; }, []() { return 3; }},
                                {[]() { return 4; }, []() { return 5; }}};
            return m;
        };
        ASSERT_THROW(make_fmatrix(), std::exception);
    }
}

TEST_F(fmatrix_test_fixture, copy) {
    {
        const fmatrix<int()> m1 = {{[]() { return 1234; }, []() { return 1234; }, []() { return 1234; }},
                                   {[]() { return 1234; }, []() { return 1234; }, []() { return 1234; }}};
        fmatrix<int()> m2;
        do {
            m2 = m1;
            ASSERT_EQ(2, m2.rows());
            ASSERT_EQ(3, m2.cols());
            for (size_t r = 0; r < m2.rows(); r++) {
                for (size_t c = 0; c < m2.cols(); c++) {
                    EXPECT_EQ(1234, m2[r][c]());
                }
            }
        } while (false);
    }
}

TEST_F(fmatrix_test_fixture, move) {
    {
        const fmatrix<int()> m1 = transpose(M1);
        fmatrix<int()> m2;
        m2 = M1 * m1;
        ASSERT_EQ(M1S.rows(), m2.rows());
        ASSERT_EQ(M1S.cols(), m2.cols());
        for (size_t r = 0; r < m2.rows(); r++) {
            for (size_t c = 0; c < m2.cols(); c++) {
                EXPECT_EQ(M1S[r][c](), m2[r][c]());
            }
        }
    }
}

TEST_F(fmatrix_test_fixture, transpose) {
    fmatrix<int()> m1 = {{[]() { return 1; }, []() { return 2; }, []() { return 3; }},
                         {[]() { return 4; }, []() { return 5; }, []() { return 6; }}};
    fmatrix<int()> m2 = transpose(m1);
    ASSERT_EQ(2, m2.cols());
    ASSERT_EQ(3, m2.rows());
    int v = 1;
    for (size_t c = 0; c < m2.cols(); c++) {
        for (size_t r = 0; r < m2.rows(); r++) {
            EXPECT_EQ(v, m2[r][c]());
            v++;
        }
    }
}

TEST_F(fmatrix_test_fixture, multiplication) {
    { ASSERT_THROW(M1 * M1, std::exception); }
    {
        fmatrix<int()> m1 = transpose(M1);
        ASSERT_NO_THROW(M1 * m1);
        auto m2 = M1 * m1;
        ASSERT_EQ(M1.rows(), m2.rows());
        ASSERT_EQ(M1.rows(), m2.cols());
        EXPECT_EQ(M1S(), m2());
    }
}

TEST_F(fmatrix_test_fixture, swap) {
    fmatrix<int()> m1 = {{[]() { return 1; }, []() { return 2; }, []() { return 3; }},
                         {[]() { return 4; }, []() { return 5; }, []() { return 6; }}};
    const fmatrix<int()> m2 = {{[]() { return 4; }, []() { return 5; }, []() { return 6; }},
                               {[]() { return 1; }, []() { return 2; }, []() { return 3; }}};
    m1.swap_row(0, 1);
    EXPECT_EQ(m2(), m1());
}

TEST_F(fmatrix_test_fixture, addition) {
    {
        fmatrix<int()> m1{2, 3};
        fmatrix<int()> m2{1, 2};
        ASSERT_THROW(m1 + m2, std::exception);
    }
    {
        fmatrix<int()> m1 = M1 + M1;
        for (size_t i = 0; i < M1.rows(); i++) {
            for (size_t j = 0; j < M1.cols(); j++) {
                EXPECT_EQ(M1[i][j]() * 2, m1[i][j]());
            }
        }
    }
}

TEST_F(fmatrix_test_fixture, subtraction) {
    {
        fmatrix<int()> m1{2, 3};
        fmatrix<int()> m2{1, 2};
        ASSERT_THROW(m1 - m2, std::exception);
    }
    {
        fmatrix<int()> m1 = M1 - M1;
        for (size_t i = 0; i < M1.rows(); i++) {
            for (size_t j = 0; j < M1.cols(); j++) {
                EXPECT_EQ(0, m1[i][j]());
            }
        }
    }
}

}  // namespace mathlib
