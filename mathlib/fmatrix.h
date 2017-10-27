/*
    "Matrix" for functional objects.
    (c) 2017 Aleksey Khozin.
*/

#ifndef MATHLIB_FMATRIX_H
#define MATHLIB_FMATRIX_H

#include "matrix.h"
#include "helpers.h"

#include <functional>

namespace mathlib {

template<typename F>
class fmatrix;

template<typename R, typename... Args>
class fmatrix<R(Args...)> {
  static_assert(is_same_helper<R, Args...>::value == true, "Different types are not allowed.");
  using function_t = std::function<R(Args...)>;
  using data_container = std::vector<typename function_t>;
public:
  fmatrix() : fmatrix(0, 0) {}
  fmatrix(const size_t r) : fmatrix(r, 1) {}
  fmatrix(const size_t r, const size_t c) : rows_(r), cols_(c) {
    data_.reserve(r * c);
  }

  matrix<R> operator ()(Args... /*args*/) {
    return matrix<R>();
  }

  size_t cols() const { return cols_; }  // number of columns
  size_t rows() const { return rows_; }  // number of rows

  bool empty() const noexcept { return data_.empty(); }

private:
  size_t rows_;
  size_t cols_;
  data_container data_;  // Array of matrix numbers
};

//typename fmatrix<R, Args...> make_fmatrix() {
//  return fmatrix<R, Args...>();
//}

}  // namespace mathlib

#endif  // MATHLIB_FMATRIX_H
