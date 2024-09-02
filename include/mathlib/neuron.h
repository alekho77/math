/*
  Object "Neuron" for artificial neural network modeling.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_NEURON_H
#define MATHLIB_NEURON_H

#include "helpers.h"

#include <cmath>

namespace mathlib {

template <typename T> struct SIGMOID {
    T operator()(T v) const {
        return T(1) / (T(1) + std::exp(-v));
    }
    static T deriv(T y) {
        return y * (1 - y);
    }
};

template <typename T> struct BIAS {
    T bias_ = T(0);
};

template <typename T> struct NOBIAS {
    static constexpr T bias_ = T(0);
};

template <typename T, size_t N, typename B = BIAS<T>, typename F = SIGMOID<T>> class neuron : private B {
    static_assert(std::is_floating_point<T>::value == true,
                  "Artificial neuron can be built only on floating point numbers.");
    static_assert(N > 0, "Number of AN synapses shall be greater than 0.");

 public:
    using value_t = T;
    static constexpr bool use_bias = std::is_same<BIAS<T>, B>::value;
    static constexpr size_t num_synapses = N;
    using weights_t = typename make_tuple_type<T, N>::type;

    neuron() {
        initialize([](size_t) { return T(1); });
    }

    template <typename G> inline void initialize(G gen) {
        initializer(gen, std::make_index_sequence<N>());
    }

    template <typename... Args> inline T operator()(Args&&... args) const {
        static_assert(sizeof...(Args) == N, "Number of arguments must be equal synapses.");
        return actfun_(combiner<N>(std::forward_as_tuple(args...)));
    }

    void set_bias(T b) {
        static_assert(use_bias, "use_bias is false!");
        if constexpr (use_bias) {
            this->bias_ = b;
        }
    }
    inline T bias() const {
        return this->bias_;
    }

    template <typename... Args> inline void set_weights(Args&&... args) {
        static_assert(sizeof...(Args) == N, "Number of arguments must be equal synapses.");
        weights_ = std::forward_as_tuple(args...);
    }
    inline const weights_t& weights() const {
        return weights_;
    }

    template <size_t I> inline void set_weight(T val) {
        std::get<I>(weights_) = val;
    }
    template <size_t I> inline T weight() const {
        return std::get<I>(weights_);
    }

    inline static T deriv(T y) {
        return F::deriv(y);
    }

 private:
    template <size_t I> inline T combiner(const weights_t& inputs) const {
        if constexpr (I == 0) {
            return this->bias_;
        } else {
            return std::get<I - 1>(inputs) * std::get<I - 1>(weights_) + combiner<I - 1>(inputs);
        }
    }

    template <typename G, size_t... I> inline void initializer(const G& gen, std::index_sequence<I...>) {
        weights_ = std::forward_as_tuple(gen(I)...);
    }

    F actfun_;
    weights_t weights_;
};

} // namespace mathlib

#endif // MATHLIB_NEURON_H
