#include <stdio.h>
#define N 100000

__device__ __managed__ int arr[N];

__global__ void foo() {
    if (threadIdx.x < N) {
        arr[threadIdx.x] = threadIdx.x * threadIdx.x;
    }
}

int main() {

    int numBlocks = 1;
    int numThreads = 128;
    foo<<<numBlocks, numThreads>>>();
    cudaDeviceSynchronize();
    return 0;
}