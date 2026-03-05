#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "gaussian.cuh"

void createGaussianKernel(std::vector<float>& kernel,
                          int kernelSize,
                          float sigma)
{
    int radius = kernelSize / 2;
    float sum = 0.0f;

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float value = expf(-(x * x + y * y) /
                               (2.0f * sigma * sigma));
            kernel[(y + radius) * kernelSize + (x + radius)] = value;
            sum += value;
        }
    }

    for (int i = 0; i < kernelSize * kernelSize; i++)
        kernel[i] /= sum;
}

int main()
{
    cv::Mat img = cv::imread("image.png",
                             cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Failed to load image\n";
        return -1;
    }

    int width = img.cols;
    int height = img.rows;

    size_t bytes = width * height * sizeof(unsigned char);

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, img.data, bytes,
               cudaMemcpyHostToDevice);

    int kernelSize = 15;
    float sigma = 4.0f;

    std::vector<float> h_kernel(kernelSize * kernelSize);
    createGaussianKernel(h_kernel, kernelSize, sigma);

    launchGaussianBlur(
        d_input,
        d_output,
        width,
        height,
        h_kernel.data(),
        kernelSize
    );

    cudaMemcpy(img.data, d_output, bytes,
               cudaMemcpyDeviceToHost);
    cv::imwrite("blurred.png", img);

    cv::imshow("Gaussian Blur CUDA", img);
    cv::waitKey(0);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}