#include "math/cnn/cntrainer.h"

#include <gtest/gtest.h>

namespace cnn {

class cntrainer_test_fixture : public ::testing::Test {
 protected:
    void SetUp() override {
        network.set_weights(0, 0, std::vector<double>{0.45, -0.12});
        network.set_weights(0, 1, std::vector<double>{0.78, 0.13});
        network.set_weights(1, 0, std::vector<double>{1.5, -2.3});
    }

    const size_t inputs = 2;
    const cnneuron neuron1 = {cnfunction::sigmoid, 2, true};
    const cnneuron neuron2 = {cnfunction::sigmoid, 2, false};
    const cnlayer hidden_layer = {cnnode{neuron2, {0, 1}}, cnnode{neuron2, {0, 1}}};

    cnnetwork network = cnnetwork(inputs, {hidden_layer, {cnnode{neuron1, {0, 1}}}});
};

TEST_F(cntrainer_test_fixture, randomizer) {
    std::mt19937 gen(1977);
    std::uniform_real_distribution<double> dis(-10, 10);
    double values[7];
    for (size_t i = 0; i < 7; i++) {
        values[i] = dis(gen);
    }

    auto trainer = cntrainer(network);
    // trainer.randomize(10, 1977);
}

}  // namespace cnn
