#include "cnnetwork.h"

#include <CL/cl.hpp>

#include <list>
#include <algorithm>

namespace mathlib {

class cnnetwork::impl {
public:
  impl();

  void add_input_layer(size_t inputs);
  void add_layer(const cnlayer& layer);

private:
  struct cllayer {
    cl_int weights_stride;
    const cl::Buffer& inputs;
    cl::Buffer nodes;  // neurons description
    cl::Buffer weights;
    cl::Buffer outputs;
  };

  using clnode = cl_int3;  // x -type, y -number of synapses, z -use bias

  cl::Context context_ = cl::Context::getDefault();
  cl::Buffer input_buf_;
  std::list<cllayer> layers_;
};

cnnetwork::cnnetwork(size_t inputs, std::initializer_list<cnlayer>&& layers)
  : impl_(std::make_unique<cnnetwork::impl>()) {
  impl_->add_input_layer(inputs);
  for (auto&& l: layers) {
    impl_->add_layer(l);
  }
}

cnnetwork::~cnnetwork() = default;

cnnetwork::impl::impl() {
  const auto devs = context_.getInfo<CL_CONTEXT_DEVICES>();
  if (devs.size() == 1) {
    const auto& dev = devs.front();
    if (!(dev.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU) || !dev.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>()) {
      throw std::logic_error("OpenCL device is unsuitable");
    }
  } else {
    throw std::logic_error("OpenCL device has not been found");
  }
}

void cnnetwork::impl::add_input_layer(size_t inputs) {
  cl::Buffer buf(context_, CL_MEM_READ_ONLY, sizeof(cl_double) * inputs);
  std::swap(input_buf_, buf);
}

void cnnetwork::impl::add_layer(const cnlayer& layer) {
  // Finding maximum number of synapses
  size_t max_w = 0;
  for (const auto& n: layer) {
    if (!n.neuron.synapses) {
      throw std::logic_error("There should be synapses");
    }
    max_w = std::max(max_w, n.neuron.synapses + 1);  // the bias is counted always
  }
  // Finding the nearest power 2 that is greater than the number
  int count = 0;
  for (; max_w > 0; count++) {
    max_w >>= 1;
  }
  layers_.emplace_back(layers_.size() > 0 ? layers_.back().outputs : input_buf_,
                       cl::Buffer(context_, CL_MEM_READ_ONLY, sizeof(clnode) * layer.size()),
                       );
}

}  // namespace mathlib
