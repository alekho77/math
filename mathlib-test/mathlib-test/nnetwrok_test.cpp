#include "math/mathlib/nnetwork.h"

#include <gtest/gtest.h>

namespace mathlib {

class nnetwork_test_fixture : public ::testing::Test {
protected:
};

TEST_F(nnetwork_test_fixture, construct) {
  using InputLayer = input_layer<double, 3>;
  using Neuron1 = neuron<double, 3>;
  using Neuron2 = neuron<double, 2>;
  using OutputLayer = std::tuple<Neuron2, Neuron1, Neuron2>;
  
  using map1 = index_pack<0, 2>;
  using map2 = index_pack<0, 1, 2>;
  using map3 = index_pack<2, 0>;
  using Map = type_pack<map1, map2, map3>;
  
  nnetwork<InputLayer, OutputLayer, Map> network;
  auto res = network(-1, 0, 1);
  EXPECT_EQ(std::make_tuple(0, 0, 0), res);
}

}  // namespace mathlib
