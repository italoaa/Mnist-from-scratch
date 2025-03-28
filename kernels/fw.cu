// input of (bs, n) matrix representing bs amount of samples where each sample has n dimentions.
__global__ void forward(int bs, int n, int out_w,
			float* input, float* weights, float* biases, float* out) {
  // y for rows (height of the mat)
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  // x for column (width of the mat)
  int column = blockIdx.x * blockDim.x + threadIdx.x; 

  // do the dot product between the row and col
  if (row < bs && col < out_w) {
    output[row*out_w + column] = biases[column];
    for (int i = 0; i < n; i++) {
      output[row * out_w + column] += weights[i * out_w + column] * input[row * n + i]
    }
  }
}

__global__ void relu(int w, int h, float* input, float* output) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int column = blockIdx.x * blockDim.x + threadIdx.x; 
  if (row < h && column < w) {
    float act = input[row * w + column];
    output[row * w + column] = act > 0.f ? act : 0.f; // relu part
  }
}

__global__ void softmax(int w, int h, float* input, float* output) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int column = blockIdx.x * blockDim.x + threadIdx.x; 
  if (row < h && column < w) {
    float maxin = input[row * w + 0];
    for (int i = 1; i < w; i++) {
      maxin = max(maxin, input[row * w + i]);
    }
    float div = 0.f;
    for (int i = 0; i < w; i++) {
      div += exp(input[row * w + i] - maxin);
    }
    output[row * w + column] = exp(input[row * w + column] - maxin) / div;
  }
}

// gt for groud truth
// input is a matrix of batch_size x num_classes
// the kernel loops of the the number of classes per item in the batch
__global__ void cross_entropy(int w, int h, float* preds, float* gt, float* output) { 
  int idx = blockIdx.x*blockDim.x + threadIdx.x; // get the index of the current thread
  if (idx < h) {
    float loss = 0.f;
    fot (int i = 0; i < w; i++) { // loop over the number of classes
      loss -= gt[idx * w + i] * log(max(1e-6, preds[idx * w + i]));
    }
    outputs[idx] = loss;
  }
}

__global__ void he_init(int w, int h, float* weights) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int column = blockIdx.x * blockDim.x + threadIdx.x; 
  if (row < h && column < w) {
    curandState state; // State for the random number generator
    curand_init(42, row * w + column, 0, &state); // Initialize the state
    weights[row * w + column] = sqrtf(2.0 / w) * curand_normal(&state);
  }
}
