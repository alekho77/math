/*
  Common mathematics helpers.
  (c) 2017 Aleksey Khozin.
*/

#ifndef MATHLIB_HELPERS_H
#define MATHLIB_HELPERS_H

#include <limits>
#include <type_traits>

namespace mathlib {

template<typename First, typename... Rest>
struct is_floating_point_helper {
  static constexpr bool value = std::is_floating_point<First>::value && is_floating_point_helper<Rest...>::value;
};

template<typename First>
struct is_floating_point_helper<First> {
  static constexpr bool value = std::is_floating_point<First>::value;
};

template<typename A, typename... Rest>
struct is_same_helper;

template<typename A, typename B, typename... Rest>
struct is_same_helper<A, B, Rest...> {
  static constexpr bool value = std::is_same<A, B>::value && is_same_helper<B, Rest...>::value;
};

template<typename A, typename B>
struct is_same_helper<A, B> {
  static constexpr bool value = std::is_same<A, B>::value;
};

template<typename A>
struct is_same_helper<A> {
  static constexpr bool value = true;
};

template<typename T>
struct numeric_helper;

template<>
struct numeric_helper<float> {
  static constexpr float epsilon = 1e-3f;
  static constexpr float step = 1e-2f;
};

template<>
struct numeric_helper<double> {
  static constexpr double epsilon = 1e-12;
  static constexpr double step = 1e-4;
};

template<>
struct numeric_helper<long double> {
  static constexpr long double epsilon = 1e-12L;
  static constexpr long double step = 1e-5L;
};

}  // namespace mathlib

#endif  // MATHLIB_HELPERS_H
