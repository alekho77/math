/*
  Object "Neuron" for artificial neural network modeling.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_NEURON_H
#define MATHLIB_NEURON_H

#include "helpers.h"

#include <utility>
#include <cmath>

namespace mathlib {

template <typename T>
struct SLOPE {
  T slope = 1;
};

template <typename T>
struct NOSLOPE {
  static constexpr T slope = 1;
};

template <typename T, typename S = NOSLOPE<T>>
struct sigmoid : S {
  T operator () (T v) {
    return T(2) / (T(1) + std::exp(-S::slope * v)) - T(1);
  }
};

template <typename T>
struct BIAS {
  T bias = 0;
};

template <typename T>
struct NOBIAS {
  static constexpr T bias = 0;
};

template <typename T, size_t N, typename B = BIAS<T>, typename F = sigmoid<T>>
class neuron {
  static_assert(std::is_floating_point<T>::value == true, "Artificial neuron can be built only on floating point numbers.");
  static_assert(N > 0, "Number of AN synapses shall be greater than 0.");

  static constexpr bool use_bias = std::is_same<BIAS<T>, B>::value;
  static constexpr bool use_slope = std::is_base_of<SLOPE<T>, F>::value;
public:
  template <typename... Args>
  T operator ()(Args&&... args) {
    static_assert(sizeof...(Args) == N, "Number of arguments must be equal synapses.");
    //{std::forward<Args>(args)...}
    return T();
  }
private:
};

}  // namespace mathlib

#endif  // MATHLIB_NEURON_H
