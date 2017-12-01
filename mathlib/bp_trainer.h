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
  using apply_t = std::function<typename Network::value_t ()>;
public:
  explicit bp_trainer(Network& net) : network_(net) {}

  void randomize(typename Network::value_t range = 10, unsigned seed = std::random_device()()) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<typename Network::value_t> dis(-range, range);
    pass_layer<Network::num_layers - 1>([&gen, &dis]() { return dis(gen); });
  }

private:
  template <size_t K>
  void pass_layer(const apply_t& fun) {
    using Layer = network_layer_t<K, Network>;
    (pass_neuron<std::tuple_size<Layer>::value - 1, Layer>(*this))(network_.layer<K>(), fun);
    pass_layer<K - 1>(fun);
  }
  template <>
  void pass_layer<0>(const apply_t& fun) {
    using Layer = network_layer_t<0, Network>;
    (pass_neuron<std::tuple_size<Layer>::value - 1, Layer>(*this))(network_.layer<0>(), fun);
  }

  template <size_t I, typename Layer>
  struct pass_neuron {
    explicit pass_neuron(bp_trainer& trainer) : trainer_(trainer) {}
    void operator ()(Layer& layer, const apply_t& fun) {
      using Neuron = std::tuple_element_t<I, Layer>;
      (apply_neuron<Neuron, Neuron::use_bias>(trainer_))(std::get<I>(layer), fun);
      (pass_neuron<I - 1, Layer>(trainer_))(layer, fun);
    }
    bp_trainer& trainer_;
  };
  template <typename Layer>
  struct pass_neuron<0, Layer> {
    explicit pass_neuron(bp_trainer& trainer) : trainer_(trainer) {}
    void operator ()(Layer& layer, const apply_t& fun) {
      using Neuron = std::tuple_element_t<0, Layer>;
      (apply_neuron<Neuron, Neuron::use_bias>(trainer_))(std::get<0>(layer), fun);
    }
    bp_trainer& trainer_;
  };

  template <typename Neuron, bool use_bias>
  struct apply_neuron {
    explicit apply_neuron(bp_trainer& trainer) : trainer_(trainer) {}
    void operator ()(Neuron& n, const apply_t& fun) {
      apply_weight(n, fun, std::make_index_sequence<Neuron::num_synapses>());
      n.set_bias(fun());
    }
    template <size_t... J>
    void apply_weight(Neuron& n, const apply_t& fun, std::index_sequence<J...>) {
      n.set_weights(weight<J>(fun)...);
    }
    template <size_t J>
    inline typename Neuron::value_t weight(const apply_t& fun) {
      return fun();
    }
    bp_trainer& trainer_;
  };
  template <typename Neuron>
  struct apply_neuron<Neuron, false> {
    explicit apply_neuron(bp_trainer& trainer) : trainer_(trainer) {}
    void operator ()(Neuron& n, const apply_t& fun) {
      apply_weight(n, fun, std::make_index_sequence<Neuron::num_synapses>());
    }
    template <size_t... J>
    void apply_weight(Neuron& n, const apply_t& fun, std::index_sequence<J...>) {
      n.set_weights(weight<J>(fun)...);
    }
    template <size_t J>
    inline typename Neuron::value_t weight(const apply_t& fun) {
      return fun();
    }
    bp_trainer& trainer_;
  };

  Network& network_;
};

template <typename Network>
static inline bp_trainer<Network> make_bp_trainer(Network& network) {
  return bp_trainer<Network>(network);
}

}  // namespace mathlib

#endif  // MATHLIB_BP_TRAINER_H
