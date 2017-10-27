/*
  Numerical differentiation.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_DIFF_H
#define MATHLIB_DIFF_H

#include <limits>
#include <functional>
#include <type_traits>
#include <tuple>

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

// Derivative
template<typename R, typename... Args>
class derivative {
  static_assert(is_floating_point_helper<R, Args...>::value == true, "Differentiating of non-floating number function is not supported.");
  static_assert(is_same_helper<R, Args...>::value == true, "Different types are not allowed.");
  using function_t = std::function<R(Args...)>;
public:
  derivative(const function_t& f, R eps) : func_(f), epsilon_(eps) {}
  explicit derivative(const function_t& f) : derivative(f, std::numeric_limits<R>::epsilon() * (R)(1000)) {}
  derivative(const derivative&) = default;
  derivative(derivative&&) = default;

  // Calculate partial derivative
  template<size_t K>
  R diff(Args... args) {
    static_assert(K < sizeof...(Args), "Index of variable has to be less than number of arguments.");
    using namespace std;
    const auto vars = make_tuple(args...);
    R h = (get<K>(vars) == (R)(0)) ? (epsilon_ * 1000) : (abs(get<K>(vars)) * epsilon_ * 1000);
    R d_res = dh<K>(h, vars, index_sequence_for<Args...>{});
    R d_eps;
    do {
      R d1 = d_res;
      h /= (R)(2);
      d_res = dh<K>(h, vars, index_sequence_for<Args...>{});
      d_eps = (d_res == (R)(0)) ? abs(d1 - d_res) : (abs(d1 - d_res) / d_res);
    } while (d_eps > epsilon_);
    return d_res;
  }
private:
  template<size_t K, size_t... I>
  R dh(R h, const std::tuple<Args...>& vars, std::index_sequence<I...>) {
    using namespace std;
    auto vars_left = vars;
    get<K>(vars_left) -= h;
    R f_left = func_(get<I>(vars_left)...);
    auto vars_right = vars;
    get<K>(vars_right) += h;
    R f_right = func_(get<I>(vars_right)...);
    return (f_right - f_left) / (2 * h);
  }
  const R epsilon_;
  function_t func_;
};

template<typename R, typename... Args>
typename derivative<R, Args...> make_deriv(R(*func)(Args...), R eps = (std::numeric_limits<R>::epsilon() * (R)(1000))) {
  return derivative<R, Args...>(func, eps);
}

template<typename C, typename R, typename... Args>
typename derivative<R, Args...> make_deriv(R(C::*func)(Args...), C* that, R eps = (std::numeric_limits<R>::epsilon() * (R)(1000))) {
  return derivative<R, Args...>([that, func](Args... args)->R { return (that->*func)(args...); }, eps);
}

}  // namespace mathlib

#endif  // MATHLIB_DIFF_H
