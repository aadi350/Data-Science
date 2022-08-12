#include <stdio.h>


__global__ void initMethod(float *a, int N) {
    /*
        Initializes an array a 
    */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i+= stride) {
        a[i] = i;
    }
}

__global__ void gpuSquare(float *result, float *a, int N) {
    /*
        Computes square of values in a on GPU
    */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        result[i] = a[i] * a[i];
    }
}


int main() {

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(
        &numberOfSMs,
        cudaDevAttrMultiProcessorCount,
        deviceId
    );


    const int N = 2<<24; // large number 
    size_t size = N * sizeof(float);

    float *a;
    float *result;

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&result, size);

    // cudaMemPrefetchAsync(a, size, deviceId);
    // cudaMemPrefetchAsync(result, size, deviceId);

    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;
    
    // initMethod<<<numberOfBlocks, threadsPerBlock>>>(a, N);
    // gpuSquare<<<numberOfBlocks, threadsPerBlock>>>(result, a, N);

    initMethod<<<1, 1>>>(a, N);
    gpuSquare<<<1, 1>>>(result, a, N);

    /* Move back to CPU */
    // cudaMemPrefetchAsync(result, size, cudaCpuDeviceId);

    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(result);



    return 0;
}