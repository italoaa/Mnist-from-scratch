// kernels from ./kenels/bw.cu and ./kernels/fw.cu
#include "../kernels/bw.cu"
#include "../kernels/fw.cu"

#include "helpers.cpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream> 
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cassert>
#include <string>

#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

int main(int argc, char** argv)
{

// Dataset sizes
int testSetSize = 10000;   // Number of test examples
int trainSetSize = 60000;  // Number of training examples

// Input and output dimensions
int inputFeatureSize = 784;  // 28x28 = 784 pixels per image
int numClasses = 10;         // 10 possible digits (0-9)

// Container arrays for the MNIST dataset
// Shape: [numExamples x inputFeatureSize] for images (flattened 28x28 pixels)
// Shape: [numExamples x numClasses] for labels (one-hot encoded)
float* trainImages = new float[inputFeatureSize * trainSetSize]; 
float* trainLabels = new float[numClasses * trainSetSize];
  
float* testImages = new float[inputFeatureSize * testSetSize];
float* testLabels = new float[numClasses * testSetSize];

// Load MNIST dataset
{
  Timer t("read mnist");
  read_mnist("./mnist_train.csv", trainSetSize, trainImages, trainLabels);
  read_mnist("./mnist_test.csv", testSetSize, testImages, testLabels);
}

// Memory 
float* deviceInput;    // Shape: [batchSize x inputFeatureSize]
float* deviceLabels;   // Shape: [batchSize x numClasses]

/// NETWORK
// Layer 1
int hiddenSize1 = 300;
float* hiddenWeights1;  // Shape: [hiddenSize1 x inputFeatureSize]
float* hiddenBiases1;   // Shape: [hiddenSize1]
float* hiddenGradients1; // Shape: [hiddenSize1 x batchSize]

// layer 2
int hiddenSize2 = 100;
float* hiddenWeights2;  // Shape: [hiddenSize2 x hiddenSize1]
float* hiddenBiases2;   // Shape: [hiddenSize2]
float* hiddenGradients2; // Shape: [hiddenSize2 x batchSize]

// Output layer: 10 neurons
int outputSize = 10;
float* outputWeights;  // Shape: [outputSize x hiddenSize2]
float* outputBiases;   // Shape: [outputSize]
float* outputGradients; // Shape: [outputSize x batchSize]

// Training hyperparameters
int BLOCK_SIZE = 16;     // Size of CUDA thread blocks
int BATCH_SIZE = 16;     // Number of examples processed in parallel
int EPOCHS = 10;         // Number of complete passes through the training set
float LEARNING_RATE = 0.003f;  // Step size for gradient descent updates


dim3 dimGrid;
dim3 dimBlock;

float* hostOutputs = new float[BATCH_SIZE * outputSize];
float* hostLosses = new float[BATCH_SIZE];

// Layer activation storage
// For each layer, we need to store:
// - preActivations (x): the weighted sum before activation function
// - activations (a): the output after applying the activation function

// Hidden layer 1 intermediates
float* preActivationsLayer1;  // Shape: [hiddenSize1 x batchSize]
float* activationsLayer1;     // Shape: [hiddenSize1 x batchSize]
  
// Hidden layer 2 intermediates
float* preActivationsLayer2;  // Shape: [hiddenSize2 x batchSize]
float* activationsLayer2;     // Shape: [hiddenSize2 x batchSize]
  
// Output layer intermediates
float* preActivationsOutput;  // Shape: [outputSize x batchSize]
float* activationsOutput;     // Shape: [outputSize x batchSize]

// Loss values for the batch
float* deviceLosses;  // Shape: [batchSize]

// Allocate and initialize GPU memory
{
  Timer init("initialization");
	
  // Allocate memory for input data
  gpuErrchk(cudaMalloc((void**) &deviceInput, inputFeatureSize * BATCH_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &deviceLabels, numClasses * BATCH_SIZE * sizeof(float)));

  // Allocate and initialize first hidden layer
  gpuErrchk(cudaMalloc((void**) &hiddenWeights1, hiddenSize1 * inputFeatureSize * sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &hiddenBiases1, hiddenSize1 * sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &hiddenGradients1, hiddenSize1 * BATCH_SIZE * sizeof(float)));
  initLayer(hiddenWeights1, hiddenBiases1, hiddenSize1, inputFeatureSize, BLOCK_SIZE);

  // Allocate and initialize second hidden layer
  gpuErrchk(cudaMalloc((void**) &hiddenWeights2, hiddenSize2 * hiddenSize1 * sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &hiddenBiases2, hiddenSize2 * sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &hiddenGradients2, hiddenSize2 * BATCH_SIZE * sizeof(float)));
  initLayer(hiddenWeights2, hiddenBiases2, hiddenSize2, hiddenSize1, BLOCK_SIZE);

  // Allocate and initialize output layer
  gpuErrchk(cudaMalloc((void**) &outputWeights, outputSize * hiddenSize2 * sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &outputBiases, outputSize * sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &outputGradients, outputSize * BATCH_SIZE * sizeof(float)));
  initLayer(outputWeights, outputBiases, outputSize, hiddenSize2, BLOCK_SIZE);

  // Allocate memory for layer activations
  gpuErrchk(cudaMalloc((void**) &preActivationsLayer1, hiddenSize1 * BATCH_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &activationsLayer1, hiddenSize1 * BATCH_SIZE * sizeof(float)));

  gpuErrchk(cudaMalloc((void**) &preActivationsLayer2, hiddenSize2 * BATCH_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &activationsLayer2, hiddenSize2 * BATCH_SIZE * sizeof(float)));

  gpuErrchk(cudaMalloc((void**) &preActivationsOutput, outputSize * BATCH_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc((void**) &activationsOutput, outputSize * BATCH_SIZE * sizeof(float)));

  gpuErrchk(cudaMalloc((void**) &deviceLosses, BATCH_SIZE * sizeof(float)));
}

float totalTrainingTime = 0.0f;

// Training loop
for(int epoch = 0; epoch < EPOCHS; epoch++) {
  float cumulativeLoss = 0.0f;
  int correctPredictions = 0;
  int totalPredictions = 0;
  auto startTime = std::chrono::system_clock::now();

  // Process mini-batches
  for(int batch = 0; batch < trainSetSize / BATCH_SIZE; batch++) {
    totalPredictions += BATCH_SIZE;
    // Copy current batch to GPU
    gpuErrchk(cudaMemcpy(
			   deviceInput, 
			   &trainImages[batch * BATCH_SIZE * inputFeatureSize], 
			   BATCH_SIZE * inputFeatureSize * sizeof(float), 
			   cudaMemcpyHostToDevice
			   )); 
    
    gpuErrchk(cudaMemcpy(
			   deviceLabels, 
			   &trainLabels[batch * BATCH_SIZE * numClasses], 
			   BATCH_SIZE * numClasses * sizeof(float), 
			   cudaMemcpyHostToDevice
			   ));

// ========== FORWARD PASS ==========
	    
// First hidden layer forward pass
// 19 , 1
dimGrid = dim3(ceil(hiddenSize1/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	    
forward<<<dimGrid, dimBlock>>>(
				     BATCH_SIZE, inputFeatureSize, hiddenSize1, 
				     deviceInput, hiddenWeights1, hiddenBiases1, preActivationsLayer1
				     );
gpuErrchk(cudaPeekAtLastError());

// Apply ReLU activation to first hidden layer
relu<<<dimGrid, dimBlock>>>(
				  hiddenSize1, BATCH_SIZE, 
				  preActivationsLayer1, activationsLayer1
				  );
gpuErrchk(cudaPeekAtLastError());

// Second hidden layer forward pass
// 100/16 = 7, (16/16)= 1  ===> 7,1
dimGrid = dim3(ceil(hiddenSize2/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	    
forward<<<dimGrid, dimBlock>>>(
				     BATCH_SIZE, hiddenSize1, hiddenSize2, 
				     activationsLayer1, hiddenWeights2, hiddenBiases2, preActivationsLayer2
				     );
gpuErrchk(cudaPeekAtLastError());


// Apply ReLU activation to second hidden layer
relu<<<dimGrid, dimBlock>>>(
				  hiddenSize2, BATCH_SIZE, 
				  preActivationsLayer2, activationsLayer2
				  );
gpuErrchk(cudaPeekAtLastError());

// Output layer forward pass
dimGrid = dim3(ceil(outputSize/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE), 1);
dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	    
forward<<<dimGrid, dimBlock>>>(
				     BATCH_SIZE, hiddenSize2, outputSize, 
				     activationsLayer2, outputWeights, outputBiases, preActivationsOutput
				     );
gpuErrchk(cudaPeekAtLastError());

// Apply softmax activation to output layer
softmax<<<dimGrid, dimBlock>>>(
				     outputSize, BATCH_SIZE, 
				     preActivationsOutput, activationsOutput
				     );
gpuErrchk(cudaPeekAtLastError());

// Compute the loss
dimGrid = dim3(ceil(outputSize/(float)BLOCK_SIZE), 1, 1);
dimBlock = dim3(BLOCK_SIZE, 1, 1);
cross_entropy<<<dimGrid, dimBlock>>>(
					   outputSize, BATCH_SIZE, 
					   activationsOutput, deviceLabels, deviceLosses
					   );

gpuErrchk(cudaDeviceSynchronize());

// Copy results back to host for evaluation
gpuErrchk(cudaMemcpy(
			   hostOutputs, activationsOutput, 
			   BATCH_SIZE * outputSize * sizeof(float), 
			   cudaMemcpyDeviceToHost
			   ));
gpuErrchk(cudaMemcpy(
			   hostLosses, deviceLosses, 
			   BATCH_SIZE * sizeof(float), 
			   cudaMemcpyDeviceToHost
			   ));

// Evaluate predictions and calculate metrics
for (int i = 0; i < BATCH_SIZE; i++) {
	float maxPredictedProb = 0.0f;
	float maxTrueProb = 0.0f;
	int predictedDigit = 0;
	int trueDigit = 0;
		
	// Find the predicted and true digits (highest probability)
	for (int j = 0; j < numClasses; j++) {
	  // Check prediction
	  if (hostOutputs[i * numClasses + j] > maxPredictedProb) {
	    maxPredictedProb = hostOutputs[i * numClasses + j];
	    predictedDigit = j;
	  }
		    
	  // Check ground truth
	  if (trainLabels[batch * BATCH_SIZE * numClasses + i * numClasses + j] > maxTrueProb) {
	    maxTrueProb = trainLabels[batch * BATCH_SIZE * numClasses + i * numClasses + j];
	    trueDigit = j;
	  }
	}
}

// Backwards
// We use the same variable to apply backprop on the inputs
// first we apply backprop from the next layer to this layer before activation
// then we apply the relu to backprop to pre activation. both are held in hiddenGradientsN variables
// it is the gradients for the layer before activation

// grads for output
dimGrid = dim3(ceil(outputSize/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE, 1));
dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

ce_back<<dimGrid, dimBlock>>(
				   outputSize, BATCH_SIZE,
				   activationsOutput, deviceLabels, outputGradients
				   );
gpuErrchk(cudaPeekAtLastError());

// backprop to second layer
dimGrid = dim3(ceil(hiddenSize2/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE, 1));
dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

// progate to the gradients before activation
backward<<dimGrid, dimBlock>>(
				    BATCH_SIZE, outputSize, hiddenSize2,
				    outputWeights, outputBiases, outputGradients, hiddenGradients2
				    );
gpuErrchk(cudaPeekAtLastError());

// back prop through the relu
relu_backwards<<<dimGrid, dimBlock>>>(
					    hiddenSize2, BATCH_SIZE, 
					    activationsLayer2, hiddenGradients2, hiddenGradients2
					    );

// backprop to first layer
dimGrid = dim3(ceil(hiddenSize1/(float)BLOCK_SIZE), ceil(BATCH_SIZE/(float)BLOCK_SIZE, 1));
dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

// progate to the gradients before activation
backward<<dimGrid, dimBlock>>(
				    BATCH_SIZE, hiddenSize2, hiddenSize1,
				    hiddenWeights2, hiddenBiases2, hiddenGradients2, hiddenGradients1
				    );
gpuErrchk(cudaPeekAtLastError());

// now through relu to pre act
relu_backwards<<<dimGrid, dimBlock>>>(
					    hiddenSize1, BATCH_SIZE, 
					    activationsLayer1, hiddenGradients1, hiddenGradients1
					    );

// Update of weights
// all the hiddenGradtiensN contain the gradients of the inputs pre activation with respect to loss
// outsize, hidsize2 = shape of weights
// we update the weights of the output layer
// we use the activations of the previous layer to update this
dimGrid = dim3(ceil(outputSize/(float)BLOCK_SIZE), ceil(hiddenSize2/(float)BLOCK_SIZE), 1);
dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
update_layer<<dimGrid, dimBlock>>(
					outputSize, hiddenSize2, BATCH_SIZE,
					LEARNING_RATE,
					outputWeights, outputBiases, activationsLayer2,
					outputGradients
					);

dimGrid = dim3(ceil(hiddenSize2/(float)BLOCK_SIZE), ceil(hiddenSize1/(float)BLOCK_SIZE), 1);
dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
update_layer<<dimGrid, dimBlock>>(
					hiddenSize2, hiddenSize3, BATCH_SIZE,
					LEARNING_RATE,
					hiddenWeights2, hiddenBiases2, activationsLayer1,
					hiddenGradients2
					);

dimGrid = dim3(ceil(hiddenSize1/(float)BLOCK_SIZE), ceil(inputFeatureSize/(float)BLOCK_SIZE), 1);
dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
update_layer<<dimGrid, dimBlock>>(
					hiddenSize1, inputFeatureSize, BATCH_SIZE,
					LEARNING_RATE,
					hiddenWeights1, hiddenBiases1, activationsLayer1,
					hiddenGradients1
					);

} // end of mini batch

// We validate the model at the end of the training loop

} // end of epoch loop

}
