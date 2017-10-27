/*
    "Matrix" for functional objects.
    (c) 2017 Aleksey Khozin.
*/

#ifndef MATHLIB_FMATRIX_H
#define MATHLIB_FMATRIX_H

#include "matrix.h"
#include "helpers.h"

namespace mathlib {

template<typename R, typename... Args>
class fmatrix {
  static_assert(is_floating_point_helper<R, Args...>::value == true, "Differentiating of non-floating number function is not supported.");
  static_assert(is_same_helper<R, Args...>::value == true, "Different types are not allowed.");
  using function_t = std::function<R(Args...)>;
public:

private:

};

}  // namespace mathlib

#endif  // MATHLIB_FMATRIX_H
