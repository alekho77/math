#include "math/cnn/cntrainer.h"

#include <gtest/gtest.h>

namespace cnn {

class cntrainer_test_fixture : public ::testing::Test {
 protected:
    void SetUp() override {
        network.set_weights(0, 0, {0.45, -0.12});
        network.set_weights(0, 1, {0.78, 0.13});
        network.set_weights(1, 0, {1.5, -2.3, 0.0});
    }

    const size_t inputs = 2;
    const cnlayer hidden_layer = {2, cnfunction::sigmoid, false};
    const cnlayer output_layer = {1, cnfunction::sigmoid, true};

    cnnetwork network{inputs, {hidden_layer, output_layer}};
};  // namespace cnn

TEST_F(cntrainer_test_fixture, randomizer) {
    std::mt19937 gen(1977);
    std::uniform_real_distribution<double> dis(-10, 10);

    cntrainer trainer(network);
    trainer.randomize(10, 1977);
    EXPECT_EQ(std::vector<double>({dis(gen), dis(gen)}), network.weights(0, 0));
    EXPECT_EQ(std::vector<double>({dis(gen), dis(gen)}), network.weights(0, 1));
    EXPECT_EQ(std::vector<double>({dis(gen), dis(gen), dis(gen)}), network.weights(1, 0));
}

TEST_F(cntrainer_test_fixture, forward_pass) {
    cntrainer trainer(network);
    trainer({1, 0}, {1});

    const std::vector<std::vector<double>> expected_states = {{0.62312115069872909, 0.66964674723395157},
                                                              {0.4174971614255642}};
    for (size_t l = 0; l < network.layer_num(); l++) {
        const auto states = trainer.states(l);
        ASSERT_EQ(expected_states[l].size(), states.size());
        for (size_t n = 0; n < states.size(); n++) {
            EXPECT_DOUBLE_EQ(expected_states[l][n], states[n]);
        }
    }
}

TEST_F(cntrainer_test_fixture, back_pass_deltas) {
    cntrainer trainer(network);
    trainer({1, 0}, {1});

    const std::vector<std::vector<double>> expected_deltas = {{0.05281718211874069, -0.07341221444187529},
                                                              {0.148097277843866}};
    for (size_t l = 0; l < network.layer_num(); l++) {
        const auto deltas = trainer.deltas(l);
        ASSERT_EQ(expected_deltas[l].size(), deltas.size());
        for (size_t n = 0; n < deltas.size(); n++) {
            EXPECT_DOUBLE_EQ(expected_deltas[l][n], deltas[n]);
        }
    }
}

TEST_F(cntrainer_test_fixture, iteration) {
    cntrainer trainer(network);
    trainer.set_learning_rate(0.7);
    trainer.set_momentum(0.3);

    {
        // The first iteration
        auto errs = trainer({1, 0}, {1});
        EXPECT_DOUBLE_EQ(0.4349516726098631, std::get<0>(errs));
        EXPECT_DOUBLE_EQ(0.3674991051506382, std::get<1>(errs));

        const std::vector<double> expected_weights_1 = {0.4869720274831185, -0.12};
        const auto weights_1 = network.weights(0, 0);
        ASSERT_EQ(expected_weights_1.size(), weights_1.size());
        for (size_t i = 0; i < weights_1.size(); i++) {
            EXPECT_DOUBLE_EQ(expected_weights_1[i], weights_1[i]);
        }

        const std::vector<double> expected_weights_2 = {0.7286114498906873, 0.13};
        const auto weights_2 = network.weights(0, 1);
        ASSERT_EQ(expected_weights_2.size(), weights_2.size());
        for (size_t i = 0; i < weights_2.size(); i++) {
            EXPECT_DOUBLE_EQ(expected_weights_2[i], weights_2[i]);
        }

        const std::vector<double> expected_weights_3 = {1.56330380580478, -2.22891684915785, 0.1036680944907062};
        const auto weights_3 = network.weights(1, 0);
        ASSERT_EQ(expected_weights_3.size(), weights_3.size());
        for (size_t i = 0; i < weights_3.size(); i++) {
            EXPECT_DOUBLE_EQ(expected_weights_3[i], weights_3[i]);
        }
    }
    {
        // The second iteration
        auto errs = trainer({1, 0}, {1});
        EXPECT_DOUBLE_EQ(0.3674991051506382, std::get<0>(errs));
        EXPECT_DOUBLE_EQ(0.2857407453988994, std::get<1>(errs));

        const std::vector<double> expected_weights_1 = {0.5353970535635459, -0.12};
        const auto weights_1 = network.weights(0, 0);
        ASSERT_EQ(expected_weights_1.size(), weights_1.size());
        for (size_t i = 0; i < weights_1.size(); i++) {
            EXPECT_DOUBLE_EQ(expected_weights_1[i], weights_1[i]);
        }

        const std::vector<double> expected_weights_2 = {0.6636227394088663, 0.13};
        const auto weights_2 = network.weights(0, 1);
        ASSERT_EQ(expected_weights_2.size(), weights_2.size());
        for (size_t i = 0; i < weights_2.size(); i++) {
            EXPECT_DOUBLE_EQ(expected_weights_2[i], weights_2[i]);
        }

        const std::vector<double> expected_weights_3 = {1.645039703595705, -2.139264721821502, 0.236068941453811};
        const auto weights_3 = network.weights(1, 0);
        ASSERT_EQ(expected_weights_3.size(), weights_3.size());
        for (size_t i = 0; i < weights_3.size(); i++) {
            EXPECT_DOUBLE_EQ(expected_weights_3[i], weights_3[i]);
        }
    }
}

}  // namespace cnn
