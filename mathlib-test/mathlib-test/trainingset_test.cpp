#include "math/mathlib/bp_trainer.h"
#include "math/mathlib/trainingset.h"

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

    // using Sample = std::tuple<std::tuple<double, double>, double>;

    Network network;
    const std::vector<double> samples = {0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0};
    const std::vector<std::tuple<double, double>> errors = {std::make_tuple(0.1610515941460189, 0.1423169375538049),
                                                            std::make_tuple(0.4678944794333771, 0.409865052384153),
                                                            std::make_tuple(0.3817107887266827, 0.3126116795194077),
                                                            std::make_tuple(0.1583207898742187, 0.1416541662911197)};
    const std::vector<std::tuple<double, double>> epoch_errors = {
        std::make_tuple(0.2922444130450744, 0.2516119589371213),
        std::make_tuple(0.2849186824051443, 0.2426129589742493),
        std::make_tuple(0.2814832227899324, 0.2384869744280807)};
};

TEST_F(trainingset_test_fixture, load) {
    using namespace boost::iostreams;
    stream_buffer<array_source> buf(reinterpret_cast<const char*>(samples.data()), sizeof(double) * samples.size());
    std::istream in(&buf);
    auto training = make_training_set<bp_trainer>(network);
    size_t count = training.load(in);
    EXPECT_EQ(4, count);
}

TEST_F(trainingset_test_fixture, training) {
    using namespace boost::iostreams;
    stream_buffer<array_source> buf(reinterpret_cast<const char*>(samples.data()), sizeof(double) * samples.size());
    std::istream in(&buf);
    auto training = make_training_set<bp_trainer>(network);
    ASSERT_EQ(4, training.load(in));
    size_t count = 0;
    auto cb = [&count, this](size_t idx, auto inputs, auto outputs, auto errs) {
        EXPECT_EQ(count, idx);
        EXPECT_EQ(this->samples[3 * idx + 0], std::get<0>(inputs));
        EXPECT_EQ(this->samples[3 * idx + 1], std::get<1>(inputs));
        EXPECT_EQ(this->samples[3 * idx + 2], std::get<0>(outputs));
        EXPECT_NEAR(std::get<0>(this->errors[idx]), std::get<0>(errs), 1e-15);
        EXPECT_NEAR(std::get<1>(this->errors[idx]), std::get<1>(errs), 1e-15);
        count++;
    };
    training.set_learning_rate(0.7);
    training.set_momentum(0.3);
    auto errs1 = training(cb);
    EXPECT_EQ(epoch_errors[0], errs1);
    auto errs2 = training([](size_t, auto, auto, auto) {});
    EXPECT_DOUBLE_EQ(std::get<0>(epoch_errors[1]), std::get<0>(errs2));
    EXPECT_DOUBLE_EQ(std::get<1>(epoch_errors[1]), std::get<1>(errs2));
    auto errs3 = training([](size_t, auto, auto, auto) {});
    EXPECT_EQ(epoch_errors[2], errs3);
}

} // namespace mathlib
