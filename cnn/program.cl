double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

__kernel void neuron_atom(const int stride,
    __global const double* inputs,
    __global const double* weights,
    __global double* inters,
    __global double* outputs,
    ) {
    const int neuron_id = get_global_id(0);
    const int weight_id = get_global_id(1);
    const int idx = neuron_id * stride + weight_id;

    inters[idx] = inputs[weight_id] * weights[idx];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = 2; i <= stride; i *= 2) {
        if ((weight_id % i) != 0) {
            return;
        }

        inters[idx] += inters[idx + i / 2];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // until here, only weight_id=0 can reach
    outputs[neuron_id] = sigmoid(inters[idx]);
}
