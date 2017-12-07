/*
  Training set for learning of artificial neural network.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_TRAININGSET_H
#define MATHLIB_TRAININGSET_H

#include "nnetwork.h"

#include <istream>
#include <vector>
#include <random>
#include <algorithm>

namespace mathlib {

template <template <typename> class Trainer, typename Network>
class training_set {
  using trainer_t = Trainer<Network>;
  using value_t = typename Network::value_t;
  using input_t = typename trainer_t::input_t;
  using output_t = typename Network::output_t;
  using sample_t = std::tuple<input_t, output_t>;
  using errors_t = std::tuple<value_t, value_t>;

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
      sort();
      return samples_.size();
    }
    return 0;
  }

  void sort() {
    std::vector<size_t> indexes;
    indexes.reserve(samples_.size());
    for (size_t i = 0; i < samples_.size(); i++) {
      indexes.emplace_back(i);
    }
    indexes_.swap(indexes);
  }

  void shuffle(unsigned seed = std::random_device()()) {
    std::mt19937 gen(seed);
    std::shuffle(indexes_.begin(), indexes_.end(), gen);
  }

  value_t learning_rate() const { return trainer_.learning_rate(); }
  void set_learning_rate(value_t eta) { trainer_.set_learning_rate(eta); }

  value_t momentum() const { return trainer_.momentum(); }
  void set_momentum(value_t alpha) { trainer_.set_momentum(alpha); }

  template <typename Callback>
  void operator ()(const Callback& cb) {
    for (size_t idx: indexes_) {
      errors_t errs = trainer_(std::get<0>(samples_[idx]), std::get<1>(samples_[idx]));
      cb(idx, std::get<0>(samples_[idx]), std::get<1>(samples_[idx]), errs);
    }
  }

private:
  Network& network_;
  trainer_t trainer_;
  std::vector<sample_t> samples_;
  std::vector<size_t> indexes_;
};

template <template <typename> class Trainer, typename Network>
static inline training_set<Trainer, Network> make_training_set(Network& net) {
  return training_set<Trainer, Network>(net);
}

}  // namespace mathlib

#endif  // MATHLIB_TRAININGSET_H
