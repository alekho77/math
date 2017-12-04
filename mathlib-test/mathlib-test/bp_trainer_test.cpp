#include "math/mathlib/bp_trainer.h"

#include <gtest/gtest.h>

namespace mathlib {

class bp_trainer_test_fixture : public ::testing::Test {
protected:
  void SetUp() override {
    auto& layer1 = network.layer<0>();
    std::get<0>(layer1).set_weights(0.45, -0.12);
    std::get<1>(layer1).set_weights(0.78, 0.13);
    auto& layer2 = network.layer<1>();
    std::get<0>(layer2).set_weights(1.5, -2.3);
  }

  using InputLayer = input_layer<double, 2>;
  using Neuron1 = neuron<double, 2>;
  using Neuron2 = neuron<double, 2, NOBIAS<double>>;
  using IndexPack = index_pack<0, 1>;
  using Map1 = make_type_pack<IndexPack, 2>::type;
  using Map2 = type_pack<IndexPack>;
  using HiddenLayer = nnetwork<InputLayer, std::tuple<Neuron2, Neuron2>, Map1>;
  using Network = nnetwork<HiddenLayer, std::tuple<Neuron1>, Map2>;

  Network network;
};

TEST_F(bp_trainer_test_fixture, randomizer) {
  std::mt19937 gen(1977);
  std::uniform_real_distribution<double> dis(-10, 10);
  double values[7];
  for (size_t i = 0; i < 7; i++) {
    values[i] = dis(gen);
  }

  auto trainer = make_bp_trainer(network);
  trainer.randomize(10, 1977);

  const auto& layer1 = network.layer<0>();
  const auto& layer2 = network.layer<1>();
  EXPECT_EQ(std::make_tuple(values[0], values[1]), std::get<0>(layer1).weights());
  EXPECT_EQ(std::make_tuple(values[2], values[3]), std::get<1>(layer1).weights());
  EXPECT_EQ(std::make_tuple(values[4], values[5]), std::get<0>(layer2).weights());
  EXPECT_EQ(values[6], std::get<0>(layer2).bias());
}

TEST_F(bp_trainer_test_fixture, forward_pass) {
  const auto trainer = make_bp_trainer(network);
  const auto net_state = trainer.states(std::make_tuple(1, 0));
  const auto expected_state = std::make_tuple(std::make_tuple(0.2212784678984441, 0.3713602278765078), std::make_tuple(-0.2553291700256212));
  static_assert(std::is_same<decltype(expected_state), decltype(net_state)>::value, "Unexpected result type.");
  EXPECT_DOUBLE_EQ(std::get<0>(std::get<0>(expected_state)), std::get<0>(std::get<0>(net_state)));
  EXPECT_DOUBLE_EQ(std::get<1>(std::get<0>(expected_state)), std::get<1>(std::get<0>(net_state)));
  EXPECT_DOUBLE_EQ(std::get<0>(std::get<1>(expected_state)), std::get<0>(std::get<1>(net_state)));
}

TEST_F(bp_trainer_test_fixture, back_pass_1) {
  const auto trainer = make_bp_trainer(network);
  const auto deltas = trainer.deltas(trainer.states(std::make_tuple(1, 0)), std::make_tuple(1));
  const auto expected_deltas = std::make_tuple(std::make_tuple(0.4185118261795359, -0.5817023683861284), std::make_tuple(0.5867452570956306));
  static_assert(std::is_same<decltype(expected_deltas), decltype(deltas)>::value, "Unexpected result type.");
//   EXPECT_DOUBLE_EQ(std::get<0>(std::get<0>(expected_state)), std::get<0>(std::get<0>(net_state)));
//   EXPECT_DOUBLE_EQ(std::get<1>(std::get<0>(expected_state)), std::get<1>(std::get<0>(net_state)));
//   EXPECT_DOUBLE_EQ(std::get<0>(std::get<1>(expected_state)), std::get<0>(std::get<1>(net_state)));
}

}  // namespace mathlib
