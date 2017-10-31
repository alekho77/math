/*
  Numerical approximation.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_APPROX_H
#define MATHLIB_APPROX_H

#include "matrix.h"
#include "lsyseq.h"
#include "helpers.h"

#include <vector>
#include <memory>
#include <tuple>

namespace mathlib {

template<typename T, size_t N>
class approx {
  static_assert(N > 0, "Must be at least one variable.");
public:
  // Add new approach, the last arg is the const term (b).
  template <typename... Args>
  approx& operator ()(Args... args) {
    static_assert(sizeof...(Args) == (N + 1), "Number of arguments must be one more than variables.");
    static_assert(are_same<T, Args...>::value == true, "All types must be the same.");
    add_approach({args...});
    return *this;
  }

  matrix<T> approach() {
    matrix<T> A{approaches_.size(), N};
    matrix<T> B{approaches_.size()};
    for (size_t i = 0; i < approaches_.size(); i++) {
      matrix_row(i, A, B, approaches_[i], std::make_index_sequence<N>());
    }
    matrix<T> AT = transpose(A);
    linear_equations<T> syseq{AT * A, AT * B};
    return syseq.normalize().solve();
  }
private:
  template <size_t I>
  struct TypeForIdx { typedef T Type; };

  template <size_t... I>
  static auto helper(std::index_sequence<I...>) -> decltype(std::make_tuple(TypeForIdx<I>::Type()...)) {}

  using pack_t = decltype(helper(std::make_index_sequence<N + 1>()));

  void add_approach(pack_t&& data) {
    approaches_.push_back(data);
  }

  template <size_t... I>
  void matrix_row(size_t idx, matrix<T>& A, matrix<T>& B, const pack_t& pack, std::index_sequence<I...>) {
    A[idx][I] = std::get<I>(pack)...;
    B[idx][0] = std::get<N>(pack);
  }

  std::vector<pack_t> approaches_;
};

}  // namespace mathlib

#endif  // MATHLIB_DERIVATIVE_H
