#include "math/mathlib/cntrainer.h"

#include <gtest/gtest.h>

namespace mathlib {

class cntrainer_test_fixture : public ::testing::Test {
 protected:
    void SetUp() override {
        // auto& layer1 = network.layer<0>();
        // std::get<0>(layer1).set_weights(0.45, -0.12);
        // std::get<1>(layer1).set_weights(0.78, 0.13);
        // auto& layer2 = network.layer<1>();
        // std::get<0>(layer2).set_weights(1.5, -2.3);
    }

    const size_t inputs = 2;
    const cnneuron neuron1 = {cnfunction::sigmoid, 2, true};
    const cnneuron neuron2 = {cnfunction::sigmoid, 2, false};
    const cnlayer hidden_layer = {cnnode{neuron2, {0, 1}}, cnnode{neuron2, {0, 1}}};

    cnnetwork network = cnnetwork(inputs, {hidden_layer, {cnnode{neuron1, {0, 1}}}});
};

}  // namespace mathlib
