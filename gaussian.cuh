#ifndef GAUSSIAN_CUH
#define GAUSSIAN_CUH

#include <cuda_runtime.h>

void launchGaussianBlur(
    unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    const float* h_kernel,
    int kernelSize
);

#endif