#include "math/mathlib/cnnetwork.h"

#include <gtest/gtest.h>

namespace mathlib {

  class cnnetwork_test_fixture : public ::testing::Test {
  protected:
    //using InputLayer = input_layer<double, 3>;
    //using Neuron1 = neuron<double, 3>;
    //using Neuron2 = neuron<double, 2>;
    //using OutputLayer = std::tuple<Neuron2, Neuron1, Neuron2>;

    //using map1 = index_pack<0, 2>;
    //using map2 = index_pack<0, 1, 2>;
    //using map3 = index_pack<2, 0>;
    //using Map = type_pack<map1, map2, map3>;

    //using network1_t = nnetwork<InputLayer, OutputLayer, Map>;
    //static_assert(network1_t::num_layers == 1, "Wrong layers number");
    //static_assert(std::is_same<network_layer_t<0, network1_t>, OutputLayer>::value, "Wrong layer type");

    //using network2_t = nnetwork<network1_t, std::tuple<Neuron1>, type_pack<index_pack<1, 2, 0>>>;
    //static_assert(network2_t::num_layers == 2, "Wrong layers number");
    //static_assert(std::is_same<network_layer_t<0, network2_t>, OutputLayer>::value, "Wrong layer type");
    //static_assert(std::is_same<network_layer_t<1, network2_t>, std::tuple<Neuron1>>::value, "Wrong layer type");
  };

  TEST_F(cnnetwork_test_fixture, construct) {
    {
      cnnetwork network1(3);
      //auto res1 = network1(-1, 0, 1);

      //EXPECT_EQ(std::make_tuple(0.5, 0.5, 0.5), res1);

      //const auto& layer1 = network1.layer<0>();

      //EXPECT_EQ(0, std::get<0>(layer1).bias());
      //EXPECT_EQ(0, std::get<1>(layer1).bias());
      //EXPECT_EQ(0, std::get<2>(layer1).bias());

      //EXPECT_EQ(std::make_tuple(1, 1), std::get<0>(layer1).weights());
      //EXPECT_EQ(std::make_tuple(1, 1, 1), std::get<1>(layer1).weights());
      //EXPECT_EQ(std::make_tuple(1, 1), std::get<2>(layer1).weights());
    }
    {
      //const network2_t network2;
      //auto res2 = network2(1, 0, -1);

      //EXPECT_EQ(std::make_tuple(SIGMOID<double>()(3 * 0.5)), res2);

      //const auto& layer1 = network2.layer<0>();

      //EXPECT_EQ(0, std::get<0>(layer1).bias());
      //EXPECT_EQ(0, std::get<1>(layer1).bias());
      //EXPECT_EQ(0, std::get<2>(layer1).bias());

      //EXPECT_EQ(std::make_tuple(1, 1), std::get<0>(layer1).weights());
      //EXPECT_EQ(std::make_tuple(1, 1, 1), std::get<1>(layer1).weights());
      //EXPECT_EQ(std::make_tuple(1, 1), std::get<2>(layer1).weights());

      //const auto& layer2 = network2.layer<1>();

      //EXPECT_EQ(0, std::get<0>(layer2).bias());
      //EXPECT_EQ(std::make_tuple(1, 1, 1), std::get<0>(layer2).weights());
    }
  }

  TEST_F(cnnetwork_test_fixture, topology) {
    {
      //network1_t network1;
      //auto res1 = network1(-1, 0, 1);
      //EXPECT_EQ(std::make_tuple(0.5, 0.5, 0.5), res1);

      //auto& layer1 = network1.layer<0>();
      //std::get<0>(layer1).set_weights(-3, 1);
      //std::get<1>(layer1).set_weights(-1, -1, 1);
      //std::get<2>(layer1).set_weights(1, -3);

      //EXPECT_EQ(std::make_tuple(0.5, 0.5, 0.5), network1(1, 2, 3));
    }
  }

}  // namespace mathlib
