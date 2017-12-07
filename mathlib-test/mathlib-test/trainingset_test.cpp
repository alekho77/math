#include "math/mathlib/trainingset.h"
#include "math/mathlib/bp_trainer.h"

#include <gtest/gtest.h>

#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream_buffer.hpp>

namespace mathlib {

class trainingset_test_fixture : public ::testing::Test {
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

  using Sample = std::tuple<std::tuple<double, double>, double>;

  Network network;
  const std::vector<Sample> samples = {
    std::make_tuple(std::make_tuple(0.0, 0.0), 0.0),
    std::make_tuple(std::make_tuple(1.0, 0.0), 1.0),
    std::make_tuple(std::make_tuple(0.0, 1.0), 1.0),
    std::make_tuple(std::make_tuple(1.0, 1.0), 0.0)
  };
  const std::vector<std::tuple<double, double>> errors = {
    std::make_tuple(0.0, 0.0),
    std::make_tuple(1.575851325117215, 0.5151895873080313),
    std::make_tuple(0.8321630367385499, 0.223760066502746),
    std::make_tuple(0.4117459150788285, 0.332333996443997)
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

TEST_F(trainingset_test_fixture, training) {
  using namespace boost::iostreams;
  stream_buffer<array_source> buf(reinterpret_cast<const char*>(samples.data()), sizeof(Sample) * samples.size());
  std::istream in(&buf);
  auto training = make_training_set<bp_trainer>(network);
  ASSERT_EQ(4, training.load(in));
  size_t count = 0;
  auto cb = [&count, this](size_t idx, auto inputs, auto outputs, auto errs) {
    EXPECT_EQ(count, idx);
    EXPECT_EQ(std::get<0>(this->samples[idx]), inputs);
    EXPECT_EQ(std::get<1>(this->samples[idx]), std::get<0>(outputs));
    EXPECT_DOUBLE_EQ(std::get<0>(this->errors[idx]), std::get<0>(errs));
    EXPECT_DOUBLE_EQ(std::get<1>(this->errors[idx]), std::get<1>(errs));
    count++;
  };
  training.set_learning_rate(0.7);
  training.set_momentum(0.3);
  training(cb);
}

}  // namespace mathlib
