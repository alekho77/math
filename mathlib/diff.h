/*
  Numerical differentiation.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_DIFF_H
#define MATHLIB_DIFF_H

#include <limits>
#include <functional>
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

template<typename A, typename B, typename... Rest>
struct is_same_helper {
  static constexpr bool value = std::is_same<A, B>::value && is_same_helper<B, Rest...>::value;
};

template<typename A, typename B>
struct is_same_helper<A, B> {
  static constexpr bool value = std::is_same<A, B>::value;
};

// Partial derivative
template<typename R, typename... Args>
class diff {
  static_assert(is_floating_point_helper<R, Args...>::value == true, "Differentiating of non-floating number function is not supported.");
  static_assert(is_same_helper<R, Args...>::value == true, "Different types are not allowed.");
  using function_t = std::function<R(Args...)>;
public:
  diff(const function_t& f, R eps) : func_(f), epsilon_(eps) {}
  explicit diff(const function_t& f) : diff(f, std::numeric_limits<R>::epsilon() * (R)(10)) {}
  diff(const diff&) = default;
  diff(diff&&) = default;

  R operator ()(Args... args) {
    return func_(args...);
  }
private:
  R epsilon_;
  function_t func_;
};

template<typename R, typename... Args>
typename diff<R, Args...> make_diff(R(*func)(Args...), R eps = (std::numeric_limits<R>::epsilon() * (R)(10))) {
  return diff<R, Args...>(func, eps);
}

template<typename C, typename R, typename... Args>
typename diff<R, Args...> make_diff(R(C::*func)(Args...), C* that, R eps = (std::numeric_limits<R>::epsilon() * (R)(10))) {
  return diff<R, Args...>([that, func](Args... args)->R { return (that->*func)(args...); }, eps);
}

}  // namespace mathlib

#endif  // MATHLIB_DIFF_H
