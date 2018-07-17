double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double deriv_sigmoid(double y) {
    return y * (1 - y);
}

__kernel void neuron_atom(__global const double* inputs,
                          __global const double* weights,
                          __global double* inters,
                          __global double* outputs) {
    const size_t stride = get_global_size(1);
    const size_t neuron_id = get_global_id(0);
    const size_t weight_id = get_global_id(1);
    const size_t idx = neuron_id * stride + weight_id;

    inters[idx] = inputs[weight_id] * weights[idx];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (size_t i = 2; i <= stride; i *= 2) {
        if ((weight_id % i) != 0) {
            return;
        }

        inters[idx] += inters[idx + i / 2];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // until here, only weight_id=0 can reach
    outputs[neuron_id] = sigmoid(inters[idx]);
}

__kernel void delta_atom(__global const double* weights_prev,
                         __global double* inters_prev,
                         __global const double* deltas_prev,
                         __global const double* outputs,
                         __global double* deltas) {
    const size_t prev_output_size = get_global_size(0);
    const size_t prev_stride = get_global_size(1);
    const size_t prev_output_id = get_global_id(0);
    const size_t prev_weight_id = get_global_id(1);
    const size_t idx = prev_output_id * prev_stride + prev_weight_id;

    inters_prev[idx] = weights_prev[idx] * deltas_prev[prev_output_id];
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    for (size_t i = 2; i <= prev_output_size; i *= 2) {
        if ((prev_output_id % i) != 0) {
            return;
        }
        
        inters_prev[idx] += inters_prev[idx + (i / 2) * prev_stride];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    
    // until here only kernels with prev_output_id=0 can reach
    deltas[prev_weight_id] = deriv_sigmoid(outputs[prev_weight_id]) * inters_prev[prev_weight_id];
}

__kernel void adjust_atom(double eta, double alpha,
                          __global const double* inputs,
                          __global const double* deltas,
                          __global double* weights,
                          __global double* adjustments) {
    const size_t stride = get_global_size(1);
    const size_t neuron_id = get_global_id(0);
    const size_t weight_id = get_global_id(1);
    const size_t idx = neuron_id * stride + weight_id;
    
    const double adj = eta * inputs[weight_id] * deltas[neuron_id] + alpha * adjustments[idx];
    weights[idx] += adj;
    adjustments[idx] = adj;
}