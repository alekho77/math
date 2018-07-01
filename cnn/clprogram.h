static constexpr char clprogram_src[] = R"(
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

__kernel void neuron_atom(__global const int3* nodes,
    __global const double* inputs,
    __global const double* weights,
    __global const int* map,
    __global double* inters,
    __global double* outputs,
    const int weights_stride) {
    const int neuron_id = get_global_id(0);
    const int weight_id = get_global_id(1);
    const int idx = neuron_id * weights_stride + weight_id;

    if (weight_id < nodes[neuron_id].y) {
        inters[idx] = inputs[map[idx]] * weights[idx + 1];
    } else {
        inters[idx] = 0;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = 2; i <= weights_stride; i *= 2) {
        if ((weight_id % i) != 0) {
            return;
        }

        inters[idx] += inters[idx + i / 2];

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // until here, only weight_id=0 can reach
    outputs[neuron_id] = sigmoid(inters[idx] + weights[idx]);  // taking into account the bias
}
)";
