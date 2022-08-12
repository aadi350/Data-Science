#include <stdio.h>

#define BLOCK_SIZE 16

/* Shared memory */
typedef struct {
  int width;
  int height;
  float* elements;
} Matrix;

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void MatMul(const Matrix A, const Matrix B, Matrix C) {
  //Load A and B to device memory
  Matrix d_A;
  d_A.width = A.width;
  d_A.height = A.height;
  
  size_t size = A.width * A.height * sizeof(float);

  cudaMalloc(&d_A.elements, size);
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

  Matrix d_B;
  d_B.width = B.width;
  d_B.height = B.height;
  size = B.width * B.height * sizeof(float);

  cudaMalloc(&d_B.elements, size);
  cudaMemcpy(d_B.elements, B.elements, size, cudaMemCpyHostToDevice);

  Matrix d_C;
  d_C.width = C.width;
  d_C.height = C.height;

  size =C.width * C.height * sizeof(float);
  cudaMalloc(&d_C.elements, size);

  /* invoke kernel */
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBloc.x, A.height / dimBlock.y);
}

int main() {
  return 0;
}
