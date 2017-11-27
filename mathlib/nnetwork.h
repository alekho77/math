/*
  Object "Neural Network" for artificial neural network modeling.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_NNETWORK_H
#define MATHLIB_NNETWORK_H

#include "neuron.h"
#include "static_indexes.h"

namespace mathlib {

template <typename T, size_t N>
struct input_layer {
  using value_t = T;
  using input_tuple = typename make_tuple_type<T, N>::tuple_type;
  static constexpr size_t input_size = N;

  template <typename... Args>
  input_tuple operator ()(Args... args) const {
    return std::forward_as_tuple(args...);
  }
};

// Recursive template to build artificial neural network.
/**
  Input - either special type of input layer model or another nnetwork.
  Output - tuple of neurons.
  Map - type pack where each element is index pack of mapping indexes for appropriate neuron.
*/
template <typename Input, typename Output, typename Map>
class nnetwork {
  static constexpr size_t output_size = std::tuple_size<Output>::value;  // Number of neurons in the output layer.
  static_assert(pack_size<Map>() == output_size, "Map shall contains mapping indexes for each output neuron.");

public:
  using value_t = typename Input::value_t;
  static constexpr size_t input_size = Input::input_size;  // Number of network input arguments.

  using output_t = typename make_tuple_type<value_t, output_size>::tuple_type;

  template <typename... Args>
  output_t operator ()(Args... args) const {
    static_assert(sizeof...(Args) == input_size, "Number of arguments must be equal input layer.");
    return mapping(input_(args...), std::make_index_sequence<output_size>());
  }

private:
  template <typename H, size_t... I>
  output_t mapping(H&& data, std::index_sequence<I...>) const {
    // data is result of input network that should be mapped on output layer inputs.
    return std::forward_as_tuple(map_neuron<I, typename get_type<Map, I>::type>(data)...);
  }

  template <size_t K, typename IdxPack, typename H>
  value_t map_neuron(const H& data) const {
    return call_neuron<IdxPack>(std::get<K>(output_), data, std::make_index_sequence<IdxPack::size>());
  }

  template <typename IdxPack, typename Neuron, typename H, size_t... W>
  value_t call_neuron(const Neuron& n, const H& data, std::index_sequence<W...>) const {
    return n(std::get<get_index<IdxPack, W>()>(data)...);
  }

  Input  input_;
  Output output_;
};

}  // namespace mathlib

#endif  // MATHLIB_NNETWORK_H
