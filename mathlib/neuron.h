/*
  Object "Neuron" for artificial neural network modeling.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_NEURON_H
#define MATHLIB_NEURON_H

#include "helpers.h"

#include <cmath>

namespace mathlib {

template <typename T>
struct SLOPE {
  T slope_ = T(1);
};

template <typename T>
struct NOSLOPE {
  static constexpr T slope_ = T(1);
};

template <typename T, typename S = NOSLOPE<T>>
struct SIGMOID : S {
  T operator () (T v) const {
    return T(2) / (T(1) + std::exp(-S::slope_ * v)) - T(1);
  }
};

template <typename T>
struct BIAS {
  T bias_ = T(0);
};

template <typename T>
struct NOBIAS {
  static constexpr T bias_ = T(0);
};

template <typename T, size_t N, typename B = BIAS<T>, typename F = SIGMOID<T>>
class neuron : private B {
  static_assert(std::is_floating_point<T>::value == true, "Artificial neuron can be built only on floating point numbers.");
  static_assert(N > 0, "Number of AN synapses shall be greater than 0.");

  static constexpr bool use_bias = std::is_same<BIAS<T>, B>::value;
  static constexpr bool use_slope = std::is_base_of<SLOPE<T>, F>::value;


public:
  static constexpr size_t num_synapses = N;
  using weights_t = typename make_tuple_type<T, N>::tuple_type;

  neuron() {
    initialize([](size_t) { return T(1); });
  }

  template <typename G>
  inline void initialize(G gen) {
    initializer(gen, std::make_index_sequence<N>());
  }

  template <typename... Args>
  inline T operator ()(Args&&... args) const {
    static_assert(sizeof...(Args) == N, "Number of arguments must be equal synapses.");
    return actfun_(combiner<N>(std::forward_as_tuple(args...)));
  }

  template <typename = std::enable_if_t<use_bias>>
  inline void set_bias(T b) {
    bias_ = b;
  }
  inline T bias() const { return bias_; }

  template <typename = std::enable_if_t<use_slope>>
  inline void set_slope(T s) {
    actfun_.slope_ = s;
  }
  inline T slope() const { return actfun_.slope_; }

  template <typename... Args>
  inline void set_weights(Args&&... args) {
    static_assert(sizeof...(Args) == N, "Number of arguments must be equal synapses.");
    weights_ = std::forward_as_tuple(args...);
  }
  inline weights_t weights() const { return weights_; }

private:
  template <size_t I>
  inline T combiner(const weights_t& inputs) const {
    return std::get<I-1>(inputs) * std::get<I-1>(weights_) + combiner<I-1>(inputs);
  }
  template <>
  inline T combiner<0>(const weights_t&) const {
    return bias_;
  }

  template <typename G, size_t... I>
  inline void initializer(const G& gen, std::index_sequence<I...>) {
    weights_ = std::forward_as_tuple(gen(I)...);
  }

  F actfun_;
  weights_t weights_;
};

}  // namespace mathlib

#endif  // MATHLIB_NEURON_H
