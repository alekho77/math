#include "math/mathlib/nnetwork.h"

#include <gtest/gtest.h>

namespace mathlib {

class nnetwork_test_fixture : public ::testing::Test {
protected:
  using InputLayer = input_layer<double, 3>;
  using Neuron1 = neuron<double, 3>;
  using Neuron2 = neuron<double, 2>;
  using OutputLayer = std::tuple<Neuron2, Neuron1, Neuron2>;

  using map1 = index_pack<0, 2>;
  using map2 = index_pack<0, 1, 2>;
  using map3 = index_pack<2, 0>;
  using Map = type_pack<map1, map2, map3>;

  using network1_t = nnetwork<InputLayer, OutputLayer, Map>;
  static_assert(network1_t::num_layers == 1, "Wrong layers number");

  using network2_t = nnetwork<network1_t, std::tuple<Neuron1>, type_pack<index_pack<1, 2, 0>>>;
  static_assert(network2_t::num_layers == 2, "Wrong layers number");
};

TEST_F(nnetwork_test_fixture, construct) {
  {
    network1_t network1;
    auto res1 = network1(-1, 0, 1);
    EXPECT_EQ(std::make_tuple(0, 0, 0), res1);
  }
  {
    const network2_t network2;
    auto res2 = network2(1, 0, -1);
    EXPECT_EQ(std::make_tuple(0), res2);
  }
}

}  // namespace mathlib
