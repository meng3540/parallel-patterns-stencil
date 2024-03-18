#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 10
#define M 10
#define BLOCK_SIZE 16

__global__ void stencilOperation(int* input, int* output, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        int index = row * width + col;
        // Perform the stencil operation
        if (row > 0 && row < N - 1 && col > 0 && col < M - 1) {
            output[index] = input[index - width] + input[index] + input[index + width] +
                input[index - 1] + input[index + 1];
        }
        else {
            output[index] = input[index];
        }
    }
}

int main() {
    int input[N][M], output[N][M];
    int* d_input, * d_output;

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            input[i][j] = i * M + j;
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, N * M * sizeof(int));
    cudaMalloc((void**)&d_output, N * M * sizeof(int));

    // Copy input array to device
    cudaMemcpy(d_input, input, N * M * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    stencilOperation << <grid, block >> > (d_input, d_output, M);

    // Copy result back to host
    cudaMemcpy(output, d_output, N * M * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Input array:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%d \t", input[i][j]);
        }
        printf("\n");
    }

    printf("Output array:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%d \t", output[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
