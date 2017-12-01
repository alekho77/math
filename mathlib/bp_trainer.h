/*
  Back propagation trainer for artificial neural network.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_BP_TRAINER_H
#define MATHLIB_BP_TRAINER_H

#include "nnetwork.h"

namespace mathlib {

template <typename Network>
class bp_trainer {
public:
  explicit bp_trainer(Network& net) : network_(net) {}

  void randomize() {
    pass_layer<Network::num_layers - 1>();
  }

private:
  template <size_t K>
  void pass_layer() {
    using Layer = network_layer_t<K, Network>;
    pass_neuron<std::tuple_size<Layer>::value - 1, Layer>(*this)(network_.layer<K>());
    pass_layer<K - 1>();
  }
  template <>
  void pass_layer<0>() {
    using Layer = network_layer_t<0, Network>;
    pass_neuron<std::tuple_size<Layer>::value - 1, Layer>(*this)(network_.layer<0>());
  }

  template <size_t I, typename Layer>
  struct pass_neuron {
    explicit pass_neuron(bp_trainer& trainer) : trainer_(trainer) {}
    void operator ()(Layer& layer) {
      using Neuron = std::tuple_element_t<I, Layer>;
      apply_neuron<Neuron, Neuron::use_bias> an(trainer_);
      an(std::get<I>(layer));
      pass_neuron<I - 1, Layer> pn(trainer_);
      pn(layer);
    }
    bp_trainer& trainer_;
  };
  template <typename Layer>
  struct pass_neuron<0, Layer> {
    explicit pass_neuron(bp_trainer& trainer) : trainer_(trainer) {}
    void operator ()(Layer& layer) {
      using Neuron = std::tuple_element_t<0, Layer>;
      apply_neuron<Neuron, Neuron::use_bias> an(trainer_);
      an(std::get<0>(layer));
    }
    bp_trainer& trainer_;
  };

  template <typename Neuron, bool use_bias>
  struct apply_neuron {
    explicit apply_neuron(bp_trainer& trainer) : trainer_(trainer) {}
    void operator ()(Neuron& n) {
      n.set_bias(0);
    }
    bp_trainer& trainer_;
  };
  template <typename Neuron>
  struct apply_neuron<Neuron, false> {
    explicit apply_neuron(bp_trainer& trainer) : trainer_(trainer) {}
    void operator ()(Neuron& /*n*/) {
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
