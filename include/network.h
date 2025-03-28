#include <vector>






#pragma once

class NN {
private:
  std::vector<int> layers;

  // host mem
  std::vector<float*> w;
  std::vector<float*> b;

  // device mem
  std::vector<float*> d_w;
  std::vector<float*> d_b;
  std::vector<float*> d_a;
  std::vector<float*> d_deltas;
