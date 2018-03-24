/*
  Numerical differentiation.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_DERIVATIVE_H
#define MATHLIB_DERIVATIVE_H

#include "helpers.h"

#include <functional>
#include <tuple>

namespace mathlib {

template <typename F>
class derivative;

// Derivative
template <typename R, typename... Args>
class derivative<R(Args...)> {
    static_assert(are_floating_points<R, Args...>::value == true,
                  "Differentiating of non-floating number function is not supported.");
    static_assert(are_same<R, Args...>::value == true, "Different types are not allowed.");
    using function_t = std::function<R(Args...)>;

 public:
    derivative() : derivative(function_t()) {}
    derivative(const function_t& f, R eps) : func_(f), epsilon_(eps) {}
    explicit derivative(const function_t& f) : derivative(f, numeric_consts<R>::epsilon) {}
    explicit derivative(function_t&& f) : derivative(std::move(f), numeric_consts<R>::epsilon) {}

    // Calculate partial derivative
    template <size_t K>
    R diff(Args... args) const {
        static_assert(K < sizeof...(Args), "Index of variable has to be less than number of arguments.");
        using namespace std;
        const auto vars = make_tuple(args...);
        R h = (get<K>(vars) == (R)(0)) ? numeric_consts<R>::step : (abs(get<K>(vars)) * numeric_consts<R>::step);
        auto d = make_dh<K>(h, vars);
        R eps = epsilon(d);
        while (eps > epsilon_) {
            h /= (R)(2);
            auto dt = make_dh<K>(h, vars);
            R epst = epsilon(dt);
            if (epst >= eps) {
                break;
            }
            d = dt;
            eps = epst;
        }
        return (get<0>(d) + get<1>(d)) / (R)(2);
    }

    R fun(Args... args) const {
        return func_(args...);
    }

 private:
    template <size_t K, size_t... I>
    R dh(const R h, const std::tuple<Args...>& vars, std::index_sequence<I...>) const {
        using namespace std;
        auto vars_left = vars;
        get<K>(vars_left) -= h;
        R f_left = func_(get<I>(vars_left)...);
        auto vars_right = vars;
        get<K>(vars_right) += h;
        R f_right = func_(get<I>(vars_right)...);
        return (f_right - f_left) / (2 * h);
    }

    template <size_t K>
    inline std::tuple<R, R> make_dh(const R h, const std::tuple<Args...>& vars) const {
        using namespace std;
        return make_tuple(dh<K>(h, vars, index_sequence_for<Args...>{}),
                          dh<K>(h / (R)(2), vars, index_sequence_for<Args...>{}));
    }

    inline R epsilon(const std::tuple<R, R>& d) const {
        using namespace std;
        return (get<0>(d) == (R)(0) || (get<1>(d) == (R)(0))) ? abs(get<0>(d) - get<1>(d))
                                                              : abs((get<0>(d) - get<1>(d)) / get<0>(d));
    }

    R epsilon_;
    function_t func_;
};

template <typename R, typename... Args>
typename derivative<R(Args...)> make_deriv(R (*func)(Args...), R eps = numeric_consts<R>::epsilon) {
    return derivative<R(Args...)>(func, eps);
}

template <typename C, typename R, typename... Args>
typename derivative<R(Args...)> make_deriv(R (C::*func)(Args...), C* that, R eps = numeric_consts<R>::epsilon) {
    return derivative<R(Args...)>([that, func](Args... args) -> R { return (that->*func)(args...); }, eps);
}

}  // namespace mathlib

#endif  // MATHLIB_DERIVATIVE_H
