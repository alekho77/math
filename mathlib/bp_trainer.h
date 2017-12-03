/*
  Back propagation trainer for artificial neural network.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_BP_TRAINER_H
#define MATHLIB_BP_TRAINER_H

#include "nnetwork.h"

#include <random>
#include <functional>

namespace mathlib {

template <typename Network>
class bp_trainer {
  using value_t = typename Network::value_t;
  using apply_t = std::function<value_t ()>;
  using input_t = typename make_tuple_type<value_t, Network::input_size>::type;

  static constexpr size_t output_size = std::tuple_size<typename Network::output_t>::value;  // Number of neurons in the output layer.

  template <size_t... I>
  static auto helper_network_data(std::index_sequence<I...>) -> decltype(
    std::make_tuple(typename make_tuple_type<value_t, std::tuple_size<network_layer_t<I, Network>>::value>::type()...)) {}
  using network_data_t = decltype(helper_network_data(std::make_index_sequence<Network::num_layers>()));

public:
  explicit bp_trainer(Network& net) : network_(net) {}

  void randomize(value_t range = 1, unsigned seed = std::random_device()()) {
    (randomizer(network_))(range, seed);
  }

  network_data_t states(const input_t& inputs) const {
    return (forward_pass(network_))(inputs);
  }

  network_data_t deltas(const network_data_t& states, const typename Network::output_t& desired_outputs) {
    
    return network_data_t();
  }

  value_t operator ()(const input_t& inputs, const typename Network::output_t& expected_outputs) {
    // Forward pass to determine values of all neurons.
    //network_data_t neurons_values = forward_pass(inputs);
    
    //auto actual_outputs = call_network(inputs, std::make_index_sequence<Network::input_size>());
    //pass_layers<Network::num_layers - 1>([&expected_outputs, &actual_outputs]() { return 0; });
    //auto fixed_outputs = call_network(inputs, std::make_index_sequence<Network::input_size>());
    return error(expected_outputs, fixed_outputs, std::make_index_sequence<output_size>());
  }

private:
  class randomizer {
  public:
    explicit randomizer(Network& net) : network_(net) {}

    inline void operator ()(value_t range, unsigned seed) {
      gen_.seed(seed);
      dis_ = std::uniform_real_distribution<value_t>(-range, range);
      walk_layer<0>();
    }

  private:
    template <size_t K>
    inline void walk_layer() {
      using Layer = network_layer_t<K, Network>;
      layer_walker<Layer>(*this)(network_.layer<K>());
      walk_layer<K + 1>();
    }
    template <>
    inline void walk_layer<Network::num_layers>() {}

    template <typename Layer>
    struct layer_walker {
      explicit layer_walker(randomizer& rand) : random_(rand) {}

      inline void operator ()(Layer& layer) {
        walk_neuron<0>(layer);
      }

      template <size_t I>
      inline void walk_neuron(Layer& layer) {
        using Neuron = std::tuple_element_t<I, Layer>;
        (neuron_walker<Neuron, Neuron::use_bias>(random_))(std::get<I>(layer));
        walk_neuron<I + 1>(layer);
      }
      template <>
      inline void walk_neuron<std::tuple_size<Layer>::value>(Layer&) {}

      randomizer& random_;
    };

    template <typename Neuron>
    struct neuron_walker_base {
      explicit neuron_walker_base(randomizer& rand) : random_(rand) {}

      template <size_t J>
      inline void walk_weight(Neuron& n) {
        n.set_weight<J>(random_.rand());
        walk_weight<J + 1>(n);
      }
      template <>
      inline void walk_weight<Neuron::num_synapses>(Neuron&) {}

      randomizer& random_;
    };

    template <typename Neuron, bool use_bias>
    struct neuron_walker : neuron_walker_base<Neuron> {
      explicit neuron_walker(randomizer& rand) : neuron_walker_base<Neuron>(rand) {}
      inline void operator ()(Neuron& n) {
        walk_weight<0>(n);
        n.set_bias(random_.rand());
      }
    };

    template <typename Neuron>
    struct neuron_walker<Neuron, false> : neuron_walker_base<Neuron> {
      explicit neuron_walker(randomizer& rand) : neuron_walker_base<Neuron>(rand) {}
      inline void operator ()(Neuron& n) {
        walk_weight<0>(n);
      }
    };

    inline value_t rand() { return dis_(gen_); }

    std::mt19937 gen_;
    std::uniform_real_distribution<value_t> dis_;
    Network& network_;
  };

  class forward_pass {
  public:
    explicit forward_pass(const Network& net) : network_(net), data_() {}

    inline network_data_t operator ()(const input_t& inputs) {
      walk_layer<0>(inputs);
      return data_;
    }

  private:
    template <size_t K>
    inline void walk_layer(const input_t& inputs) {
      using InputLayer = typename network_layer_t<K - 1, Network>;
      using Input = typename make_tuple_type<value_t, std::tuple_size<InputLayer>::value>::type;
      using Layer = network_layer_t<K, Network>;
      using Map = network_map_t<K, Network>;
      std::get<K>(data_) = layer_walker<Input, Layer, Map>()(std::get<K - 1>(data_), network_.layer<K>());
      walk_layer<K + 1>(inputs);
    }
    template <>
    inline void walk_layer<0>(const input_t& inputs) {
      using Layer = network_layer_t<0, Network>;
      using Map = network_map_t<0, Network>;
      std::get<0>(data_) = layer_walker<input_t, Layer, Map>()(inputs, network_.layer<0>());
      walk_layer<1>(inputs);
    }
    template <>
    inline void walk_layer<Network::num_layers>(const input_t&) {}

    template <typename Input, typename Layer, typename Map>
    struct layer_walker {
      using output_t = typename make_tuple_type<value_t, std::tuple_size<Layer>::value>::type;

      inline output_t operator ()(const Input& inputs, const Layer& layer) {
        return map_layer(inputs, layer, std::make_index_sequence<std::tuple_size<Layer>::value>());
      }

      template <size_t... I>
      inline output_t map_layer(const Input& inputs, const Layer& layer, std::index_sequence<I...>) {
        return std::forward_as_tuple(map_neuron<get_type_t<I, Map>>(inputs, std::get<I>(layer))...);
      }

      template <typename Indexes, typename Neuron>
      inline value_t map_neuron(const Input& inputs, const Neuron& n) {
        return call_neuron<Indexes>(inputs, n, std::make_index_sequence<Indexes::size>());
      }

      template <typename Indexes, typename Neuron, size_t... J>
      inline value_t call_neuron(const Input& inputs, const Neuron& n, std::index_sequence<J...>) {
        return n(std::get<get_index<J, Indexes>()>(inputs)...);
      }
    };

    const Network& network_;
    network_data_t data_;
  };

  template <size_t... I>
  inline value_t error(const typename Network::output_t& expected, const typename Network::output_t& actual, std::index_sequence<I...>) {
    return std::sqrt(sum_error<output_size - 1>({error_function(std::get<I>(expected), std::get<I>(actual))...}) / output_size);
  }

  template <size_t I>
  inline value_t sum_error(const typename Network::output_t& errs) {
    return std::get<I>(errs) + sum_error<I - 1>(errs);
  }

  template <>
  inline value_t sum_error<0>(const typename Network::output_t& errs) {
    return std::get<0>(errs);
  }

  inline value_t error_function(value_t expected, value_t actual) {
    return powi(expected - actual, 2);
  }

  Network& network_;
};

template <typename Network>
static inline bp_trainer<Network> make_bp_trainer(Network& network) {
  return bp_trainer<Network>(network);
}

}  // namespace mathlib

#endif  // MATHLIB_BP_TRAINER_H
