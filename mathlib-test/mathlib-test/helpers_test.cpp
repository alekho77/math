#include "helpers.h"

#include <gtest/gtest.h>

namespace mathlib {

TEST(mathlib_helpers, powi) {
    EXPECT_DOUBLE_EQ(1024.0, powi(2.0, 10));
    EXPECT_DOUBLE_EQ(1.0 / 1024.0, powi(2.0, -10));
}

} // namespace mathlib
