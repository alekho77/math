/*
  Numerical approximation.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_APPROX_H
#define MATHLIB_APPROX_H

#include "matrix.h"
#include "helpers.h"

#include <vector>
#include <memory>
#include <tuple>

namespace mathlib {

template<typename T>
struct data_holder_base {

};

template<typename T, typename... Args>
class data_holder : public data_holder_base<T> {
public:
  data_holder(T b, Args... args) : const_(b), coefs_(args...) {}
private:
  T const_;
  std::tuple<Args...> coefs_;
};

template<typename T>
class approx {
public:
  // Add new approach
  template <typename... Args>
  approx<T>& operator ()(T b, Args... args) {
    static_assert(sizeof...(Args) > 0, "Must be at least one variable.");
    static_assert(are_same<T, Args...>::value == true, "All types must be the same.");
    if (vars_count == 0) {
      vars_count = sizeof...(Args);
    } else if (vars_count != sizeof...(Args)) {
      throw std::invalid_argument("Number of variables cannot be different.");
    }
    approaches_.push_back(std::make_shared<data_holder<T, Args...>>(b, args...));
    return *this;
  }
private:
  size_t vars_count = 0;
  std::vector<std::shared_ptr<data_holder_base<T>>> approaches_;
  matrix<T> A_;
  matrix<T> B_;
};

}  // namespace mathlib

#endif  // MATHLIB_DERIVATIVE_H
