#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 10
#define M 10
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Kernel function for stencil operation
__global__ void stencilOperation(int* input, int* output, int width, int height) {
    // Thread index within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Compute global indices for the block's corner
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    // Compute global indices for the thread
    int row = by + ty;
    int col = bx + tx;

    // Perform the stencil operation
    if (row < height && col < width) {
        // Compute the sum of the neighboring elements
        int sum = input[row * width + col];
        if (row > 0)           sum += input[(row - 1) * width + col];       // Upper neighbor
        if (row < height - 1)  sum += input[(row + 1) * width + col];       // Lower neighbor
        if (col > 0)           sum += input[row * width + col - 1];         // Left neighbor
        if (col < width - 1)   sum += input[row * width + col + 1];         // Right neighbor
        // Store the result in the output array
        output[row * width + col] = sum;
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

    // Allocate device memory for input and output arrays
    cudaMalloc((void**)&d_input, N * M * sizeof(int));
    cudaMalloc((void**)&d_output, N * M * sizeof(int));

    // Copy input array from host to device
    cudaMemcpy(d_input, input, N * M * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with appropriate grid and block dimensions
    dim3 grid((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (N + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    stencilOperation << <grid, block >> > (d_input, d_output, M, N);

    // Copy result back from device to host
    cudaMemcpy(output, d_output, N * M * sizeof(int), cudaMemcpyDeviceToHost);

    // Print input array
    printf("Input array:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%d\t", input[i][j]);
        }
        printf("\n");
    }

    // Print output array
    printf("Output array:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%d\t", output[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}