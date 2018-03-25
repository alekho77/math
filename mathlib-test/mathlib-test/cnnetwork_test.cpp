#include "math/mathlib/cnnetwork.h"

#include <gtest/gtest.h>

namespace mathlib {

class cnnetwork_test_fixture : public ::testing::Test {
 protected:
    const size_t inputs = 3;
    const cnneuron neuron1 = {cnfunction::sigmoid, 3, true};
    const cnneuron neuron2 = {cnfunction::sigmoid, 2, true};
    const cnlayer layer1 = {cnnode{neuron2, {0, 2}}, cnnode{neuron1, {0, 1, 2}}, cnnode{neuron2, {2, 0}}};

    // static_assert(network1_t::num_layers == 1, "Wrong layers number");
    // static_assert(std::is_same<network_layer_t<0, network1_t>, OutputLayer>::value, "Wrong layer type");

    // using network2_t = nnetwork<network1_t, std::tuple<Neuron1>, type_pack<index_pack<1, 2, 0>>>;
    // static_assert(network2_t::num_layers == 2, "Wrong layers number");
    // static_assert(std::is_same<network_layer_t<0, network2_t>, OutputLayer>::value, "Wrong layer type");
    // static_assert(std::is_same<network_layer_t<1, network2_t>, std::tuple<Neuron1>>::value, "Wrong layer type");
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

        // auto res1 = network1(-1, 0, 1);
        // ASSERT_TRUE(res1.size() == 3);
        // EXPECT_EQ(std::make_tuple(0.5, 0.5, 0.5), res1);

        // const auto& layer1 = network1.layer<0>();

        // EXPECT_EQ(0, std::get<0>(layer1).bias());
        // EXPECT_EQ(0, std::get<1>(layer1).bias());
        // EXPECT_EQ(0, std::get<2>(layer1).bias());

        // EXPECT_EQ(std::make_tuple(1, 1), std::get<0>(layer1).weights());
        // EXPECT_EQ(std::make_tuple(1, 1, 1), std::get<1>(layer1).weights());
        // EXPECT_EQ(std::make_tuple(1, 1), std::get<2>(layer1).weights());
    }
    {
        // const network2_t network2;
        // auto res2 = network2(1, 0, -1);

        // EXPECT_EQ(std::make_tuple(SIGMOID<double>()(3 * 0.5)), res2);

        // const auto& layer1 = network2.layer<0>();

        // EXPECT_EQ(0, std::get<0>(layer1).bias());
        // EXPECT_EQ(0, std::get<1>(layer1).bias());
        // EXPECT_EQ(0, std::get<2>(layer1).bias());

        // EXPECT_EQ(std::make_tuple(1, 1), std::get<0>(layer1).weights());
        // EXPECT_EQ(std::make_tuple(1, 1, 1), std::get<1>(layer1).weights());
        // EXPECT_EQ(std::make_tuple(1, 1), std::get<2>(layer1).weights());

        // const auto& layer2 = network2.layer<1>();

        // EXPECT_EQ(0, std::get<0>(layer2).bias());
        // EXPECT_EQ(std::make_tuple(1, 1, 1), std::get<0>(layer2).weights());
    }
}

TEST_F(cnnetwork_test_fixture, topology) {
    {
        // network1_t network1;
        // auto res1 = network1(-1, 0, 1);
        // EXPECT_EQ(std::make_tuple(0.5, 0.5, 0.5), res1);

        // auto& layer1 = network1.layer<0>();
        // std::get<0>(layer1).set_weights(-3, 1);
        // std::get<1>(layer1).set_weights(-1, -1, 1);
        // std::get<2>(layer1).set_weights(1, -3);

        // EXPECT_EQ(std::make_tuple(0.5, 0.5, 0.5), network1(1, 2, 3));
    }
}

}  // namespace mathlib
