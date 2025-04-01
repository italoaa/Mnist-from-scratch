nvcc -o net src/main.cu src/helpers.cu kernels/bw.cu kernels/fw.cu && ./net
