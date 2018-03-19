/*
    Solving system of non-linear equations.
    (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_NONLSYSEQ_H
#define MATHLIB_NONLSYSEQ_H

#include "lsyseq.h"
#include "jacobian.h"

namespace mathlib {

template<typename F>
class nonlinear_equations;

template<typename R, typename... Args>
class nonlinear_equations<R(Args...)> {
  static_assert(sizeof...(Args) > 0, "Functions of the system shall have arguments.");
  using function_t = std::function<R(Args...)>;

public:
  nonlinear_equations() = delete;
  nonlinear_equations(std::initializer_list<function_t>&& list)
    : F_(transpose(fmatrix<R(Args...)>{list}))
    , W_(std::move(list)) {
    if (list.size() != sizeof...(Args)) {
      throw std::invalid_argument("Number of equations shall be equal number of variables.");
    }
  }

  matrix<R> solve(Args... args) const noexcept(false) {
    return solve(transpose(matrix<R>{{args...}}), numeric_consts<R>::epsilon);
  }

  matrix<R> solve(matrix<R> x, const R epsilon) const noexcept(false) {
    x += make_syseq(x, std::index_sequence_for<Args...>{}).normalize().solve();
    R eps = residual(1, x);
    while (eps > epsilon) {
      matrix<R> xt = x + make_syseq(x, std::index_sequence_for<Args...>{}).normalize().solve();
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
  R residual_adapter(int n, const matrix<R>& x, std::index_sequence<I...>) const {
    return residual(n, x[I][0]...);
  }

  template<size_t... I>
  linear_equations<R> make_syseq(const matrix<R>& x, std::index_sequence<I...>) const {
    using namespace std;
    auto w = W_(x[I][0]...);
    auto f = -F_(x[I][0]...);
    for (size_t i = 0; i < w.rows(); i++) {
      const R min_d = (max)(numeric_consts<R>::step, abs(f[i][0] / (numeric_consts<R>::increment * max(numeric_consts<R>::increment, abs(x[i][0])))));
      if (abs(w[i][i]) < min_d) {
        w[i][i] = copysign(min_d, w[i][i]);
      }
    }
    return linear_equations<R>(w, f);
  }

  fmatrix<R(Args...)> F_;  // Ñolumn-matrix for system of non-linear equations, it is assumed that each F_[i][0](args...) = 0.
  jacobian<R(Args...)> W_;
};

}  // namespace mathlib

#endif  // MATHLIB_NONLSYSEQ_H
