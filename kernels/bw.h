#ifndef BW
#define BW
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>


__global__ void ce_back(int w, int h, float* preds, float* gt, float* output);
__global__ void backward(int bs, int n, int out_w, float* weights, float* biases, float* d_l, float* out_d_l);

__global__ void relu_backwards(int w, int h, float* a, float* d_l, float* b);

__global__ void update_layer(int w, int h, int bs, float lr, float* weights, float* biases, float* activations, float* d_l);
#endif
