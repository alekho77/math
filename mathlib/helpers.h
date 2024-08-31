/*
  Common mathematics helpers.
  (c) 2017 Aleksey Khozin.
*/

#ifndef MATHLIB_HELPERS_H
#define MATHLIB_HELPERS_H

#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mathlib {

template <typename First, typename... Rest> struct are_floating_points {
    static constexpr bool value = std::is_floating_point<First>::value && are_floating_points<Rest...>::value;
};

template <typename First> struct are_floating_points<First> {
    static constexpr bool value = std::is_floating_point<First>::value;
};

template <typename...> struct are_same;

template <typename A, typename B, typename... Rest> struct are_same<A, B, Rest...> {
    static constexpr bool value = std::is_same<A, B>::value && are_same<B, Rest...>::value;
};

template <typename A, typename B> struct are_same<A, B> {
    static constexpr bool value = std::is_same<A, B>::value;
};

template <typename A> struct are_same<A> {
    static constexpr bool value = true;
};

template <typename T> struct numeric_consts;

template <> struct numeric_consts<float> {
    static constexpr float epsilon = 0.001f;
    static constexpr float step = 0.01f;
    static constexpr float increment = 0.1f;
};

template <> struct numeric_consts<double> {
    static constexpr double epsilon = 1e-12;
    static constexpr double step = 1e-4;
    static constexpr double increment = 0.01;
};

template <> struct numeric_consts<long double> {
    static constexpr long double epsilon = 1e-12L;
    static constexpr long double step = 1e-5L;
    static constexpr long double increment = 0.01L;
};

template <typename T> T powi(const T val, int p) {
    T res = 1;
    if (p >= 0) {
        for (int i = 0; i < p; i++) {
            res *= val;
        }
    } else {
        for (int i = p; i < 0; i++) {
            res /= val;
        }
    }
    return res;
}

template <typename T, size_t N> class make_tuple_type {
    template <size_t I> using next_type = T;
    template <size_t... I>
    static auto helper(std::index_sequence<I...>) -> decltype(std::make_tuple(next_type<I>()...)) {}

 public:
    using type = decltype(helper(std::make_index_sequence<N>()));
};

} // namespace mathlib

#endif // MATHLIB_HELPERS_H
