#include "math/mathlib/bp_trainer.h"

#include <gtest/gtest.h>

namespace mathlib {

class bp_trainer_test_fixture : public ::testing::Test {
protected:
  using InputLayer = input_layer<double, 2>;
  using Neuron1 = neuron<double, 2>;
  using Neuron2 = neuron<double, 2, NOBIAS<double>>;
  using IndexPack = index_pack<0, 1>;
  using Map1 = type_pack<IndexPack, IndexPack>;
  using Map2 = type_pack<IndexPack>;
  using HiddenLayer = nnetwork<InputLayer, std::tuple<Neuron2, Neuron2>, Map1>;
  using Network = nnetwork<HiddenLayer, std::tuple<Neuron1>, Map2>;

  Network network;
};

TEST_F(bp_trainer_test_fixture, construct) {

  auto trainer = make_bp_trainer(network);
  trainer.randomize();
}

}  // namespace mathlib
