/*
  Training set for learning of artificial neural network.
  (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_TRAININGSET_H
#define MATHLIB_TRAININGSET_H

#include "nnetwork.h"

#include <istream>

namespace mathlib {

template <template <typename> class Trainer, typename Network>
class training_set {
public:
  explicit training_set(Network& net) : network_(net), trainer_(net) {}

  size_t load(std::istream& in) {

  }

private:
  Network& network_;
  Trainer<Network> trainer_;
};

template <template <typename> class Trainer, typename Network>
static inline training_set<Trainer, Network> make_training_set(Network& net) {
  return training_set<Trainer, Network>(net);
}

}  // namespace mathlib

#endif  // MATHLIB_TRAININGSET_H
