#ifndef FW
#define FW
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void forward(int bs, int n, int out_w, float* input, float* weights, float* biases, float* output);

__global__ void relu(int w, int h, float* input, float* output);

__global__ void softmax(int w, int h, float* input, float* output);

__global__ void cross_entropy(int w, int h, float* preds, float* gt, float* output);
#endif
