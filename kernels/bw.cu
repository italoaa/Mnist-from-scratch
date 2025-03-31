#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void ce_back(int w, int h, float* preds, float* gt, float* output) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  // x for column (width of the mat)
  int column = blockIdx.x * blockDim.x + threadIdx.x; 
  if (row < h && column < w) {
    // $$\frac{\partial \mathcal{L}}{\partial w} = \hat{y} - y $$

    output[row * w + column] = preds[row * w + column] - gt[row * w + column]
   }
}

__global__ void backward(int bs, int n, int out_w, float* weights, float* biases, float* d_l, float* out_d_l) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int column = blockIdx.x * blockDim.x + threadIdx.x; 
  if (row < bs && column < n) {
    float dl = 0.f;
    // $$ \frac{\partial \mathcal{L}}{\partial x^{n-1}} = \frac{\partial \mathcal{L}}{\partial x^{n}} W^n $$
    // in english our weights times the derivative of the next layer so n + 1
    for (int i = 0; i < n; i++) {
      float w = weights[i * out_w + column];
      dl += w * d_l[row * n + i];
    }
    out_d_l[row * out_w + column] = dl;
  }
}

__global__ void relu_bw(int w, int h, int ns, float* a, float* d_l, float* b) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int column = blockIdx.x * blockDim.x + threadIdx.x; 
  if (row < bs && column < n) {
    float act = a[row * w + column];
    b[row * w + column] = act > 0.f ? d_l[row * w + column] : 0.f;
  }
}

__global__ update_layer(int w, int h, int bs, float lr, float* weights, float* biases, float* activations, float* d_l) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int column = blockIdx.x * blockDim.x + threadIdx.x; 
  if (row < bs && column < n) {
    float dw = 0.f;
    float db = 0.f;
    for (int i = 0; i < bs ; i++) {
      float act = activations[i * h + row];
      float dl = d_l[i * w + column];
      dw += act * dl;
      db += dl;
    }
    weights[row * w + column] -= lr * dw / bs;
    biases[column] -= lr * db / bs;
  }
}
