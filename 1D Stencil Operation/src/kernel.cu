#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 10
#define BLOCK_SIZE 256

__global__ void stencilOperation(const int* input, int* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        // Perform the stencil operation
        output[index] = input[index - 1] + input[index] + input[index + 1];
    }
}

int main() {
    int input[N], output[N];
    int* d_input, * d_output;

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        input[i] = i;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));

    // Copy input array to device
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stencilOperation << <numBlocks, BLOCK_SIZE >> > (d_input, d_output);

    // Copy result back to host
    cudaMemcpy(output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Input array: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", input[i]);
    }
    printf("\n");

    printf("Output array: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", output[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}