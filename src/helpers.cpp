int main() {
void print_matrix(int w, int h, float* matrix, std::string title)
{
  float* m_h = new float[w*h];
  cudaMemcpy(m_h, matrix, w*h*sizeof(float), cudaMemcpyDeviceToHost);
  std::cout<<title<<std::endl;
  for(int i = 0; i<h; i++)
  {
    for(int j = 0; j<w; j++)
    {
      std::cout<<std::fixed<<std::setprecision(3)<<m_h[i*w+j]<<", ";
    }
    std::cout<<std::endl;
  }
  free(m_h);
}

void initLayer(float* weights, float* biases, int w, int h, int BLOCK_SIZE)
{
  dim3 dimGrid = dim3(ceil(w/(float)BLOCK_SIZE), ceil(h/(float)BLOCK_SIZE), 1);
  dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  init_rand<<<dimGrid, dimBlock>>>(w, h, weights);
  gpuErrchk(cudaPeekAtLastError());

  dimGrid = dim3(ceil(h/(float)BLOCK_SIZE), 1, 1);
  dimBlock = dim3(BLOCK_SIZE, 1, 1);
  init_rand<<<dimGrid, dimBlock>>>(1, h, biases);
  gpuErrchk(cudaPeekAtLastError());
}

void read_mnist(const std::string filename, int length, float* x, float* y)
{
  int input_size = 784;
  int labels = 10;

  std::fstream fin;
  fin.open(filename);
  std::string row;
  constexpr char delim = ',';
  for(int i = 0; i<length; i++)
  {
    fin >> row;
    int pos = row.find(delim);
    int label = std::stoi(row.substr(0, pos+1));
    for(int j = 0; j<labels; j++)
    {
      y[labels*i + j] = (j==label);
    }
    row.erase(0, pos+1);
    for(int j = 0; j<input_size; j++)
    {
      pos = row.find(delim);
      if (pos == std::string::npos)
      {
        pos = row.length() - 1;
      }
      x[i*input_size+j] = std::stof(row.substr(0, pos+1)) / 255; //normalize value
      row.erase(0, pos+1);
    }
    ASSERT(row.length() == 0, "didn't parse all values in row, %d", i);
  }
}
return 0;
}
