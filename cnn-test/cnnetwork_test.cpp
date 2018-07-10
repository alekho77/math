#include "math/cnn/cnnetwork.h"
#include "math/mathlib/neuron.h"

#include <gtest/gtest.h>

namespace cnn {

class cnnetwork_test_fixture : public ::testing::Test {
 protected:
    const size_t inputs = 5;
    const cnlayer layer1 = {3, cnfunction::sigmoid, true};
    const cnlayer layer2 = {1, cnfunction::sigmoid, false};
};

TEST_F(cnnetwork_test_fixture, construct) {
    {
        cnnetwork network1(inputs, {layer1});

        ASSERT_EQ(1, network1.layer_num());
        ASSERT_EQ(inputs, network1.inputs_num());
        ASSERT_EQ(layer1.nodes, network1.layer_desc(0).nodes);

        EXPECT_EQ(std::vector<double>({0.5, 0.5, 0.5}), network1(1, 2, 3, 4, 5));
        for (size_t i = 0; i < layer1.nodes; i++) {
            EXPECT_EQ(std::vector<double>({0, 0, 0, 0, 0, 0}), network1.weights(0, i));
        }
    }
    {
        cnnetwork network2(inputs, {layer1, layer2});

        ASSERT_EQ(2, network2.layer_num());
        ASSERT_EQ(inputs, network2.inputs_num());
        ASSERT_EQ(layer1.nodes, network2.layer_desc(0).nodes);
        ASSERT_EQ(layer2.nodes, network2.layer_desc(1).nodes);

        EXPECT_EQ(std::vector<double>({0.5}), network2(1, 2, 3, 4, 5));
        for (size_t i = 0; i < layer1.nodes; i++) {
            EXPECT_EQ(std::vector<double>({0, 0, 0, 0, 0, 0}), network2.weights(0, i));
        }
        EXPECT_EQ(std::vector<double>({0, 0, 0}), network2.weights(1, 0));
    }
}

TEST_F(cnnetwork_test_fixture, topology) {
    {
        cnnetwork network1(inputs, {layer1});

        std::vector<double> res1;
        for (size_t i = 0; i < layer1.nodes; i++) {
            const double wbias = 1.0 / (i + 1);
            network1.set_weights(0, i, {0, 0, 0, 0, 0, wbias});
            res1.push_back(mathlib::SIGMOID<double>{}(wbias));
        }
        EXPECT_EQ(res1, network1(1, 2, 3, 4, 5));

        std::vector<double> res2;
        for (size_t i = 0; i < layer1.nodes; i++) {
            const double wbias = -(i + 1.0);
            network1.set_weights(0, i, {1, 1, 1, -5, 4, wbias});
            res2.push_back(mathlib::SIGMOID<double>{}(6 + wbias));
        }
        EXPECT_EQ(res2, network1(1, 2, 3, 4, 5));
    }
}

}  // namespace cnn
