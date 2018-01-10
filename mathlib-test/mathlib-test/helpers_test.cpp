#include "math/mathlib/helpers.h"

#include <gtest/gtest.h>

namespace mathlib {

TEST(mathlib_helpers, powi) {
  EXPECT_DOUBLE_EQ(1024.0, powi(2.0, 10));
  EXPECT_DOUBLE_EQ(1.0 / 1024.0, powi(2.0, -10));
}

TEST(mathlib_helpers, nearest_upper_pow2) {
  // char
  EXPECT_EQ(char(0x40), nearest_upper_pow2(char(0x25)));
  EXPECT_EQ(char(1), nearest_upper_pow2(char(0)));
  EXPECT_EQ(char(0x40), nearest_upper_pow2(char(0x40)));
  EXPECT_EQ(char(0), nearest_upper_pow2(char(-1)));
  
  // unsigned char
  EXPECT_EQ(unsigned char(1), nearest_upper_pow2(unsigned char(0)));
  EXPECT_EQ(unsigned char(0x80), nearest_upper_pow2(unsigned char(0x80)));

  // short
  EXPECT_EQ(short(0x4000), nearest_upper_pow2(short(0x25A5)));
  EXPECT_EQ(short(1), nearest_upper_pow2(short(0)));
  EXPECT_EQ(short(0x4000), nearest_upper_pow2(short(0x4000)));
  EXPECT_EQ(short(0), nearest_upper_pow2(short(-1)));

  // unsigned short
  EXPECT_EQ(unsigned short(1), nearest_upper_pow2(unsigned short(0)));
  EXPECT_EQ(unsigned short(0x8000), nearest_upper_pow2(unsigned short(0x8000)));

  // int
  EXPECT_EQ(0x40000000, nearest_upper_pow2(0x25A5A5A5));
  EXPECT_EQ(1, nearest_upper_pow2(0));
  EXPECT_EQ(1, nearest_upper_pow2(1));
  EXPECT_EQ(2, nearest_upper_pow2(2));
  EXPECT_EQ(0x40000000, nearest_upper_pow2(0x40000000));
  EXPECT_EQ(0x80000000, nearest_upper_pow2(0x80000000));
  EXPECT_EQ(0, nearest_upper_pow2(-1));

  // unsigned int
  EXPECT_EQ(unsigned(1), nearest_upper_pow2(unsigned(0)));
  EXPECT_EQ(unsigned(0x80000000), nearest_upper_pow2(unsigned(0x80000000)));

  // long long
  EXPECT_EQ(long long(0x4000000000000000), nearest_upper_pow2(long long(0x25A5A5A5A5A5A5A5)));
  EXPECT_EQ(long long(1), nearest_upper_pow2(long long(0)));
  EXPECT_EQ(long long(0x4000000000000000), nearest_upper_pow2(long long(0x4000000000000000)));
  EXPECT_EQ(long long(0x8000000000000000), nearest_upper_pow2(long long(0x8000000000000000)));
  EXPECT_EQ(long long(0), nearest_upper_pow2(long long(-1)));

  // unsigned long long
  EXPECT_EQ(unsigned long long(1), nearest_upper_pow2(unsigned long long(0)));
  EXPECT_EQ(unsigned long long(0x8000000000000000), nearest_upper_pow2(unsigned long long(0x8000000000000000)));
}

}  // namespace mathlib
