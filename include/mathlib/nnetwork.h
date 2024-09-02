/*
  Object "Neural Network" for artificial neural network modeling.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_NNETWORK_H
#define MATHLIB_NNETWORK_H

#include "mathlib/neuron.h"
#include "mathlib/static_indexes.h"

namespace mathlib {

template <typename T, size_t N> struct input_layer {
    using value_t = T;
    using input_tuple = typename make_tuple_type<T, N>::type;
    static constexpr size_t input_size = N;
    static constexpr size_t num_layers = 0;

    template <typename... Args> input_tuple operator()(Args... args) const {
        return std::forward_as_tuple(args...);
    }
};

template <size_t I, typename Network> struct network_layer;

// Recursive template to build artificial neural network.
/**
  Input - either special type of input layer model or another nnetwork.
  Output - tuple of neurons.
  Map - type pack where each element is index pack of mapping indexes for appropriate neuron.
*/
template <typename Input, typename Output, typename Map> class nnetwork {
    static constexpr size_t output_size = std::tuple_size<Output>::value; // Number of neurons in the output layer.
    static_assert(pack_size<Map>() == output_size, "Map shall contains mapping indexes for each output neuron.");

 public:
    using value_t = typename Input::value_t;
    using output_t = typename make_tuple_type<value_t, output_size>::type;
    using output_layer_t = Output;
    using input_layer_t = Input;
    using map_t = Map;

    static constexpr size_t input_size = Input::input_size;     // Number of network input arguments.
    static constexpr size_t num_layers = 1 + Input::num_layers; // Number layers that are contained in the network.

    template <typename... Args> output_t operator()(Args... args) const {
        static_assert(sizeof...(Args) == input_size, "Number of arguments must be equal input layer.");
        return mapping(input_(args...), std::make_index_sequence<output_size>());
    }

    template <size_t I> const typename network_layer<I, nnetwork>::type& layer() const {
        if constexpr (I == Input::num_layers) {
            return output_;
        } else {
            return input_.template layer<I>();
        }
    }

    template <size_t I> typename network_layer<I, nnetwork>::type& layer() {
        if constexpr (I == Input::num_layers) {
            return output_;
        } else {
            return input_.template layer<I>();
        }
    }

 private:
    template <typename H, size_t... I> output_t mapping(H&& data, std::index_sequence<I...>) const {
        // data is result of input network that should be mapped on output layer inputs.
        // take I-th output neuron and I-th index map:
        return std::forward_as_tuple(map_neuron<I, get_type_t<I, Map>>(data)...);
    }

    template <size_t I, typename IdxPack, typename H> value_t map_neuron(const H& data) const {
        // call I-th neuron with mapped data:
        static_assert(std::tuple_element_t<I, Output>::num_synapses == IdxPack::size,
                      "Index map size shall be equal synapses number.");
        return call_neuron<IdxPack>(std::get<I>(output_), data, std::make_index_sequence<IdxPack::size>());
    }

    template <typename IdxPack, typename Neuron, typename H, size_t... J>
    value_t call_neuron(const Neuron& n, const H& data, std::index_sequence<J...>) const {
        return n(std::get<get_index<J, IdxPack>()>(data)...);
    }

    Input input_;
    Output output_;
};

template <size_t D, typename Network> struct network_layer_downcount {
    using type = typename network_layer_downcount<D - 1, typename Network::input_layer_t>::type;
};

template <typename Network> struct network_layer_downcount<1, Network> {
    using type = Network;
};

template <size_t I, typename Network> struct network_layer {
    static_assert(I < Network::num_layers, "Index out if bounds");
    using type = typename network_layer_downcount<Network::num_layers - I, Network>::type::output_layer_t;
};

template <size_t I, typename Network> using network_layer_t = typename network_layer<I, Network>::type;

template <size_t I, typename Network> struct network_map {
    static_assert(I < Network::num_layers, "Index out if bounds");
    using type = typename network_layer_downcount<Network::num_layers - I, Network>::type::map_t;
};

template <size_t I, typename Network> using network_map_t = typename network_map<I, Network>::type;

} // namespace mathlib

#endif // MATHLIB_NNETWORK_H
