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

    const std::vector<std::vector<double>> expected_states = {{0.610639233949222, 0.6856801139382539},
                                                              {0.3404913400038911}};
    for (size_t l = 0; l < network.layer_num(); l++) {
        const auto states = trainer.states(l);
        ASSERT_EQ(expected_states[l].size(), states.size());
        for (size_t n = 0; n < states.size(); n++) {
            EXPECT_DOUBLE_EQ(expected_states[l][n], states[n]);
        }
    }
}

}  // namespace cnn
