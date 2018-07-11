#include "cnnetwork.h"
#include "cnnetwork_impl.h"

namespace cnn {

cnnetwork::cnnetwork(size_t inputs, const std::vector<cnlayer>& layers)
    : impl_(std::make_unique<cnnetwork_impl>(inputs, layers)) {
    if (impl_->layer_num() == 0) {
        throw std::logic_error("Neural network must have at least one layer");
    }
}

cnnetwork::~cnnetwork() = default;

size_t cnnetwork::inputs_num() const {
    return impl_->inputs_num();
}

size_t cnnetwork::layer_num() const {
    return impl_->layer_num();
}

cnlayer cnnetwork::layer_desc(size_t idx) const {
    return impl_->get_layer(idx);
}

std::vector<double> cnnetwork::weights(size_t layer, size_t neuron) const {
    return impl_->weights(layer, neuron);
}

void cnnetwork::set_weights(size_t layer, size_t neuron, const std::vector<double>& weights) {
    impl_->set_weights(layer, neuron, weights);
}

std::vector<double> cnnetwork::operator()(const std::vector<double>& inputs) {
    static_assert(std::is_same<double, cl_double>::value, "OpenCL double type does not match system double type.");
    return impl_->exec(inputs);
}

}  // namespace cnn
