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
  using input_t = typename make_tuple_type<value_t, Network::input_size>::tuple_type;

  static constexpr size_t output_size = std::tuple_size<typename Network::output_t>::value;  // Number of neurons in the output layer.

  template <size_t... I>
  static auto helper_network_data(std::index_sequence<I...>) -> decltype(
    std::make_tuple(typename make_tuple_type<value_t, std::tuple_size<network_layer_t<I, Network>>::value>::tuple_type()...)) {}
  using network_data_t = decltype(helper_network_data(std::make_index_sequence<Network::num_layers>()));

public:
  explicit bp_trainer(Network& net) : network_(net) {}

  void randomize(value_t range = 1, unsigned seed = std::random_device()()) {
    (randomizer(network_))(range, seed);
  }

  value_t operator ()(const input_t& inputs, const typename Network::output_t& expected_outputs) {
    // Forward pass to determine values of all neurons.
    network_data_t neurons_values = forward_pass(inputs);
    
    auto actual_outputs = call_network(inputs, std::make_index_sequence<Network::input_size>());
    //pass_layers<Network::num_layers - 1>([&expected_outputs, &actual_outputs]() { return 0; });
    auto fixed_outputs = call_network(inputs, std::make_index_sequence<Network::input_size>());
    return error(expected_outputs, fixed_outputs, std::make_index_sequence<output_size>());
  }

private:
  class randomizer {
  public:
    explicit randomizer(Network& net) : network_(net) {}

    void operator ()(value_t range, unsigned seed) {
      gen_.seed(seed);
      dis_ = std::uniform_real_distribution<value_t>(-range, range);
      walk_layer<0>();
    }

  private:
    template <size_t K>
    void walk_layer() {
      using Layer = network_layer_t<K, Network>;
      layer_walker<Layer>(*this)(network_.layer<K>());
      walk_layer<K + 1>();
    }
    template <>
    void walk_layer<Network::num_layers>() {}

    template <typename Layer>
    struct layer_walker {
      explicit layer_walker(randomizer& rand) : random_(rand) {}

      void operator ()(Layer& layer) {
        walk_neuron<0>(layer);
      }

      template <size_t I>
      void walk_neuron(Layer& layer) {
        using Neuron = std::tuple_element_t<I, Layer>;
        (neuron_walker<Neuron, Neuron::use_bias>(random_))(std::get<I>(layer));
        walk_neuron<I + 1>(layer);
      }
      template <>
      void walk_neuron<std::tuple_size<Layer>::value>(Layer&) {}

      randomizer& random_;
    };

    template <typename Neuron>
    struct neuron_walker_base {
      explicit neuron_walker_base(randomizer& rand) : random_(rand) {}

      template <size_t J>
      void walk_weight(Neuron& n) {
        n.set_weight<J>(random_.rand());
        walk_weight<J + 1>(n);
      }
      template <>
      void walk_weight<Neuron::num_synapses>(Neuron&) {}

      randomizer& random_;
    };

    template <typename Neuron, bool use_bias>
    struct neuron_walker : neuron_walker_base<Neuron> {
      explicit neuron_walker(randomizer& rand) : neuron_walker_base(rand) {}
      void operator ()(Neuron& n) {
        walk_weight<0>(n);
        n.set_bias(random_.rand());
      }
    };

    template <typename Neuron>
    struct neuron_walker<Neuron, false> : neuron_walker_base<Neuron> {
      explicit neuron_walker(randomizer& rand) : neuron_walker_base(rand) {}
      void operator ()(Neuron& n) {
        walk_weight<0>(n);
      }
    };

    inline value_t rand() { return dis_(gen_); }

    std::mt19937 gen_;
    std::uniform_real_distribution<value_t> dis_;
    Network& network_;
  };


  network_data_t forward_pass(const input_t& inputs) {
    network_data_t neuron_values;
    call_layer<0>(inputs, neuron_values);
    return neuron_values;
  }

  template <size_t I>
  struct layer_input {
    using type = std::tuple_element_t<I - 1, network_data_t>;
  };
  template <>
  struct layer_input<0> {
    using type = input_t;
  };

  template <size_t I>
  void call_layer(typename const layer_input<I>::type& /*inputs*/, network_data_t& data) {
    //std::get<I>(data) = net_.layer<I>();
    call_layer<I + 1>(std::get<I>(data), data);
  }

  template <>
  void call_layer<Network::num_layers - 1>(typename const layer_input<Network::num_layers - 1>::type& /*inputs*/, network_data_t& /*data*/) {
    //std::get<Network::num_layers - 1>(data) = net_.layer<Network::num_layers - 1>();
  }

  template <size_t K, size_t... I>
  std::tuple_element_t<K, network_data_t> map_neurons(const network_layer_t<K, Network>& layer, std::index_sequence<I...>) {
    std::forward_as_tuple(call_neuron<>(std::get<I>(layer),  std::make_index_sequence<std::tuple_size<network_layer_t<K, Network>>::value>())...);
  }

  template <typename Indexes, typename Neuron, typename Input, size_t... J>
  value_t call_neuron(const Neuron& n, const Input& inputs, std::index_sequence<J...>) const {
    return n(std::get<get_index<J, Indexes>()>(inputs)...);
  }

  template <size_t... I>
  inline typename Network::output_t call_network(const input_t& inputs, std::index_sequence<I...>) {
    return network_(std::get<I>(inputs)...);
  }

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
