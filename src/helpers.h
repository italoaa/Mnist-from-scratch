#ifndef HELPERS
#define HELPERS

// string
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cassert>
#include <string>
#include <fstream>

#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
  {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


class Timer
{
public:
  Timer(std::string in_name) : name(in_name)
  {
    start_time = std::chrono::system_clock::now();
  }
  ~Timer()
  {
    std::cout<<name<<" took "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count()<<" ms"<<std::endl;
  }
private:
  std::chrono::time_point<std::chrono::system_clock> start_time;
  std::string name;
};

__global__ void init_rand(int w, int h, float* weights);

void print_matrix(int w, int h, float* matrix, std::string title);

void initLayer(float* weights, float* biases, int w, int h, int BLOCK_SIZE);

void read_mnist(const std::string filename, int length, float* x, float* y);

#endif
