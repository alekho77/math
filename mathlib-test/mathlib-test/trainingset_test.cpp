#include "math/mathlib/trainingset.h"
#include "math/mathlib/bp_trainer.h"

#include <gtest/gtest.h>

#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream_buffer.hpp>

namespace mathlib {

class trainingset_test_fixture : public ::testing::Test {
protected:
  using InputLayer = input_layer<double, 2>;
  using Neuron1 = neuron<double, 2>;
  using Neuron2 = neuron<double, 2, NOBIAS<double>>;
  using IndexPack = index_pack<0, 1>;
  using Map1 = make_type_pack<IndexPack, 2>::type;
  using Map2 = type_pack<IndexPack>;
  using HiddenLayer = nnetwork<InputLayer, std::tuple<Neuron2, Neuron2>, Map1>;
  using Network = nnetwork<HiddenLayer, std::tuple<Neuron1>, Map2>;

  using Sample = std::tuple<std::tuple<double, double>, double>;

  Network network;
  const std::vector<Sample> samples = {
    std::make_tuple(std::make_tuple(0.0, 0.0), 0.0),
    std::make_tuple(std::make_tuple(1.0, 0.0), 1.0),
    std::make_tuple(std::make_tuple(0.0, 1.0), 1.0),
    std::make_tuple(std::make_tuple(1.0, 1.0), 0.0)
  };
};

TEST_F(trainingset_test_fixture, load) {
  using namespace boost::iostreams;
  stream_buffer<array_source> buf(reinterpret_cast<const char*>(samples.data()), sizeof(Sample) * samples.size());
  std::istream in(&buf);
  auto training = make_training_set<bp_trainer>(network);
  size_t count = training.load(in);
  EXPECT_EQ(4, count);
}

}  // namespace mathlib
