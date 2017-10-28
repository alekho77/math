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

private:
  template<size_t... I>
  void make_derivatives(const function_t& fun, size_t idx, std::index_sequence<I...>) {
    const derivative_t& deriv = derivative_t{fun};
    derivatives_.push_back(deriv);
    make_jacobian_row(idx, {make_derivative<I>(deriv)...});
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

  static constexpr size_t args_size = sizeof...(Args);
  fmatrix<R(Args...)> F_ = fmatrix<R(Args...)>{args_size};  // Ñolumn-matrix for system of non-linear equations, it is assumed that each F_[i][0](args...) = 0.
  fmatrix<R(Args...)> W_ = fmatrix<R(Args...)>{args_size, args_size};  // Jacobian
  std::vector<derivative_t> derivatives_;
};

}  // namespace mathlib

#endif  // MATHLIB_NONLSYSEQ_H
