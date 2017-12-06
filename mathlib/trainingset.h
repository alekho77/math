/*
  Training set for learning of artificial neural network.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_TRAININGSET_H
#define MATHLIB_TRAININGSET_H

#include "nnetwork.h"

#include <istream>
#include <vector>

namespace mathlib {

template <template <typename> class Trainer, typename Network>
class training_set {
  using trainer_t = Trainer<Network>;
  using value_t = typename Network::value_t;
  using input_t = typename trainer_t::input_t;
  using output_t = typename Network::output_t;
  using sample_t = std::tuple<input_t, output_t>;

  static_assert(sizeof(sample_t) == (std::tuple_size<input_t>::value + std::tuple_size<output_t>::value) * sizeof(value_t), "Something wrong with type size or alignment.");

public:
  explicit training_set(Network& net) : network_(net), trainer_(net) {}

  size_t load(std::istream& in) {
    std::vector<sample_t> samples;
    while (in) {
      sample_t sample;
      in.read(reinterpret_cast<char*>(&sample), sizeof(sample));
      if (in) {
        samples.push_back(sample);
      }
    }
    if (in.eof()) {
      samples_.swap(samples);
      return samples_.size();
    }
    return 0;
  }

private:
  Network& network_;
  trainer_t trainer_;
  std::vector<sample_t> samples_;
};

template <template <typename> class Trainer, typename Network>
static inline training_set<Trainer, Network> make_training_set(Network& net) {
  return training_set<Trainer, Network>(net);
}

}  // namespace mathlib

#endif  // MATHLIB_TRAININGSET_H
