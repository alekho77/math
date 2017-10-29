/*
    Solving system of non-linear equations.
    (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_NONLSYSEQ_H
#define MATHLIB_NONLSYSEQ_H

#include "fmatrix.h"
#include "lsyseq.h"
#include "derivative.h"

namespace mathlib {

template<typename F>
class nonlinear_equations;

template<typename R, typename... Args>
class nonlinear_equations<R(Args...)> {
  static_assert(sizeof...(Args) > 0, "Functions of the system shall have arguments.");
  using function_t = std::function<R(Args...)>;
  using derivative_t = derivative<R(Args...)>;

public:
  nonlinear_equations() = delete;
  nonlinear_equations(const std::initializer_list<function_t>& list) {
    if (list.size() == args_size) {
      derivatives_.reserve(args_size);
      auto fiter = list.begin();
      for (size_t i = 0; i < args_size; i++, ++fiter) {
        F_[i][0] = *fiter;
        make_derivatives(*fiter, i, std::index_sequence_for<Args...>{});
      }
    } else {
      throw std::invalid_argument("Number of equations shall be equal number of variables.");
    }
  }

  matrix<R> solve(Args... args) const noexcept(false) {
    return solve(transpose(matrix<R>{{args...}}), numeric_consts<R>::epsilon);
  }

  matrix<R> solve(matrix<R> x, const R epsilon) const noexcept(false) {
    linear_equations<R> syseq = make_syseq(x, std::index_sequence_for<Args...>{});
    matrix<R> dx = syseq.normalize().solve();
    x += dx;
    R eps = residual(1, x);
    while (eps > epsilon) {
      syseq = make_syseq(x, std::index_sequence_for<Args...>{});
      dx = syseq.normalize().solve();
      matrix<R> xt = x + dx;
      R epst = residual(1, xt);
      if (epst >= eps) {
        break;
      }
      x = xt;
      eps = epst;
    }
    return x;
  }

  R residual(int n, Args... args) const {
    matrix<R> e = F_(args...);
    R r = 0;
    for (size_t i = 0; i < e.rows(); i++) {
      r += powi(std::abs(e[i][0]), n);
    }
    return std::pow(r, static_cast<R>(1) / static_cast<R>(n));
  }

  R residual(int n, const matrix<R>& x) const {
    if (x.rows() != sizeof...(Args)) {
      throw std::invalid_argument("Number of results shall be equal number of variables.");
    }
    return residual_adapter(n, x, std::index_sequence_for<Args...>{});
  }

private:
  template<size_t... I>
  void make_derivatives(const function_t& fun, size_t idx, std::index_sequence<I...>) {
    derivatives_.push_back(derivative_t{fun});
    make_jacobian_row(idx, {make_derivative<I>(derivatives_.back())...});
  }

  template<size_t K>
  function_t make_derivative(const derivative_t& deriv) {
    return [&deriv](Args... args)->R { return deriv.diff<K>(args...); };
  }

  void make_jacobian_row(size_t idx, const std::initializer_list<function_t>& list) {
    auto fiter = list.begin();
    for (size_t j = 0; j < args_size; j++, ++fiter) {
      W_[idx][j] = *fiter;
    }
  }

  template<size_t... I>
  R residual_adapter(int n, const matrix<R>& x, std::index_sequence<I...>) const {
    return residual(n, x[I][0]...);
  }

  template<size_t... I>
  linear_equations<R> make_syseq(const matrix<R>& x, std::index_sequence<I...>) const {
    return linear_equations<R>(W_(x[I][0]...), -F_(x[I][0]...));
  }

  static constexpr size_t args_size = sizeof...(Args);
  fmatrix<R(Args...)> F_ = fmatrix<R(Args...)>{args_size};  // Ñolumn-matrix for system of non-linear equations, it is assumed that each F_[i][0](args...) = 0.
  fmatrix<R(Args...)> W_ = fmatrix<R(Args...)>{args_size, args_size};  // Jacobian
  std::vector<derivative_t> derivatives_;
};

}  // namespace mathlib

#endif  // MATHLIB_NONLSYSEQ_H
