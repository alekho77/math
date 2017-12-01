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
  std::mt19937 gen(1977);
  std::uniform_real_distribution<double> dis(-10, 10);
  auto trainer = make_bp_trainer(network);
  trainer.randomize(10, 1977);
  const auto& layer1 = network.layer<0>();
  const auto& layer2 = network.layer<1>();
  EXPECT_EQ(std::make_tuple(dis(gen), dis(gen)), std::get<0>(layer2).weights());
  EXPECT_EQ(dis(gen), std::get<0>(layer2).bias());
  EXPECT_EQ(std::make_tuple(dis(gen), dis(gen)), std::get<1>(layer1).weights());
  EXPECT_EQ(std::make_tuple(dis(gen), dis(gen)), std::get<0>(layer1).weights());
}

}  // namespace mathlib
