/*
  Non-linear approximation.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_FAPPROX_H
#define MATHLIB_FAPPROX_H

#include "helpers.h"

#include <functional>
#include <vector>

namespace mathlib {

template <typename F>
class fapprox;

template<typename R, typename... Args>
class fapprox<R(Args...)> {
  static_assert(sizeof...(Args) > 0, "Must be at least one variable.");
  static_assert(are_same<R, Args...>::value == true, "All types must be the same.");

  using function_t = std::function<R(Args...)>;

public:
  // Add new approach, the last arg is the const term (b).
  fapprox& operator ()(const function_t& fun) {
    approaches_.push_back(fun);
    return *this;
  }

private:
  std::vector<function_t> approaches_;
};

}  // namespace mathlib

#endif  // MATHLIB_FAPPROX_H
