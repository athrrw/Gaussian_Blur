#include "gaussian.cuh"
#include <cuda_runtime.h>

__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, const float* kernel, int width, int height, int kernelSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int radius = kernelSize / 2;
    float sum = 0.0f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {

            int ix = x + kx;
            int iy = y + ky;

            if (ix >= 0 && ix < width &&
                iy >= 0 && iy < height) {

                float pixel = input[iy * width + ix];
                float weight = kernel[(ky + radius) * kernelSize + (kx + radius)];

                sum += pixel * weight;
            }
        }
    }

    output[y * width + x] = static_cast<unsigned char>(sum);
}

void launchGaussianBlur(unsigned char* d_input, unsigned char* d_output, int width, int height, const float* h_kernel, int kernelSize){
    float* d_kernel;
    size_t kernelBytes = kernelSize * kernelSize * sizeof(float);

    cudaMalloc(&d_kernel, kernelBytes);
    cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    gaussianBlurKernel<<<grid, block>>>(
        d_input,
        d_output,
        d_kernel,
        width,
        height,
        kernelSize
    );

    cudaDeviceSynchronize();

    cudaFree(d_kernel);
}