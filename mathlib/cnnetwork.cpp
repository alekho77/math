#include "cnnetwork.h"

#include <CL/cl.hpp>

namespace mathlib {

class cnnetwork::impl {
public:
  void add_input_layer(size_t inputs);
private:
  cl::Context context_ = cl::Context::getDefault();
  cl::Buffer input_buf_;
};

cnnetwork::cnnetwork(size_t inputs)
  : impl_(std::make_unique<cnnetwork::impl>()) {
  impl_->add_input_layer(inputs);
}

cnnetwork::~cnnetwork() = default;

void cnnetwork::impl::add_input_layer(size_t inputs) {
  cl::Buffer buf(context_, CL_MEM_READ_ONLY, sizeof(cl_double) * inputs);
}

}  // namespace mathlib
