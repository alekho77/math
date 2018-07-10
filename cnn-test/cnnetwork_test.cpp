#include "math/cnn/cnnetwork.h"
#include "math/mathlib/neuron.h"

#include <gtest/gtest.h>

namespace cnn {

class cnnetwork_test_fixture : public ::testing::Test {
 protected:
    const size_t inputs = 3;
    const cnneuron neuron1 = {cnfunction::sigmoid, 3, true};
    const cnneuron neuron2 = {cnfunction::sigmoid, 2, true};
    const cnlayer layer1 = {cnnode{neuron2, {0, 2}}, cnnode{neuron1, {0, 1, 2}}, cnnode{neuron2, {2, 0}}};
    const cnlayer layer2 = {cnnode{neuron1, {1, 2, 0}}};
};

TEST_F(cnnetwork_test_fixture, construct) {
    {
        cnnetwork network1(inputs, {layer1});

        ASSERT_EQ(1, network1.layer_num());
        ASSERT_EQ(inputs, network1.inputs_num());
        ASSERT_EQ(layer1.size(), network1.layer_desc(0).size());
        for (size_t i = 0; i < network1.layer_desc(0).size(); i++) {
            ASSERT_EQ(layer1[i], network1.layer_desc(0)[i]);
        }

        for (int i = 0; i < 1000; i++) {
            auto res = network1(-1, 0, 1);
            ASSERT_EQ(layer1.size(), res.size());
            EXPECT_EQ((std::vector<double>{0.5, 0.5, 0.5}), res);
        }

        EXPECT_EQ(0, network1.bias(0, 0));
        EXPECT_EQ(0, network1.bias(0, 1));
        EXPECT_EQ(0, network1.bias(0, 2));

        EXPECT_EQ((std::vector<double>{1, 1}), network1.weights(0, 0));
        EXPECT_EQ((std::vector<double>{1, 1, 1}), network1.weights(0, 1));
        EXPECT_EQ((std::vector<double>{1, 1}), network1.weights(0, 2));
    }
    {
        cnnetwork network2(inputs, {layer1, layer2});

        ASSERT_EQ(2, network2.layer_num());
        ASSERT_EQ(inputs, network2.inputs_num());
        ASSERT_EQ(layer1.size(), network2.layer_desc(0).size());
        ASSERT_EQ(layer2.size(), network2.layer_desc(1).size());
        for (size_t i = 0; i < network2.layer_desc(0).size(); i++) {
            ASSERT_EQ(layer1[i], network2.layer_desc(0)[i]);
        }
        for (size_t i = 0; i < network2.layer_desc(1).size(); i++) {
            ASSERT_EQ(layer2[i], network2.layer_desc(1)[i]);
        }

        for (int i = 0; i < 1000; i++) {
            auto res = network2(1, 0, -1);
            ASSERT_EQ(layer2.size(), res.size());
            EXPECT_EQ((std::vector<double>{mathlib::SIGMOID<double>()(3 * 0.5)}), res);
        }

        EXPECT_EQ(0, network2.bias(0, 0));
        EXPECT_EQ(0, network2.bias(0, 1));
        EXPECT_EQ(0, network2.bias(0, 2));

        EXPECT_EQ((std::vector<double>{1, 1}), network2.weights(0, 0));
        EXPECT_EQ((std::vector<double>{1, 1, 1}), network2.weights(0, 1));
        EXPECT_EQ((std::vector<double>{1, 1}), network2.weights(0, 2));

        EXPECT_EQ(0, network2.bias(1, 0));
        EXPECT_EQ((std::vector<double>{1, 1, 1}), network2.weights(1, 0));
    }
}

TEST_F(cnnetwork_test_fixture, topology) {
    {
        cnnetwork network1(inputs, {layer1});

        for (int i = 0; i < 1000; i++) {
            EXPECT_EQ((std::vector<double>{0.5, 0.5, 0.5}), network1(-1, 0, 1));
        }

        network1.set_weights(0, 0, std::vector<double>{-3, 2});
        network1.set_weights(0, 1, std::vector<double>{-1, -2, 1});
        network1.set_weights(0, 2, std::vector<double>{1, -4});
        network1.set_bias(0, 0, -3);
        network1.set_bias(0, 1, 2);
        network1.set_bias(0, 2, 1);

        ASSERT_EQ(-3, network1.bias(0, 0));
        ASSERT_EQ(2, network1.bias(0, 1));
        ASSERT_EQ(1, network1.bias(0, 2));

        ASSERT_EQ((std::vector<double>{-3, 2}), network1.weights(0, 0));
        ASSERT_EQ((std::vector<double>{-1, -2, 1}), network1.weights(0, 1));
        ASSERT_EQ((std::vector<double>{1, -4}), network1.weights(0, 2));

        for (int i = 0; i < 1000; i++) {
            EXPECT_EQ((std::vector<double>{0.5, 0.5, 0.5}), network1(1, 2, 3));
        }
    }
}

}  // namespace cnn
