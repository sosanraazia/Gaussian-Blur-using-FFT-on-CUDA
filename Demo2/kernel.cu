#define _USE_MATH_DEFINES 
#include <iostream>
#include "opencv2/opencv.hpp"
#include "fftw3.h"
#include <cufft.h>


// CUDA kernel to perform element-wise multiplication of complex data without shared memory optimization
__global__ void kernelMultiply(cufftComplex* data, const cufftComplex* kernel, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if the thread is within the image bounds
	if (x < width && y < height) {
		int index = y * width + x;

		// Perform element-wise complex multiplication
		cufftComplex dataValue = data[index];
		cufftComplex kernelValue = kernel[index];
		cufftComplex result;

		result.x = dataValue.x * kernelValue.x - dataValue.y * kernelValue.y;
		result.y = dataValue.x * kernelValue.y + dataValue.y * kernelValue.x;

		// Store the result back to the data array
		data[index] = result;
	}
}

#define BLOCK_SIZE 64

// CUDA kernel to perform element-wise multiplication of complex data
__global__ void kernelMultiply_(cufftComplex* data, const cufftComplex* kernel, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Define the size of the shared memory array
	__shared__ cufftComplex sharedKernel[BLOCK_SIZE][BLOCK_SIZE]; // Adjust BLOCK_SIZE as needed

	// Check if the thread is within the image bounds
	if (x < width && y < height) {
		int index = y * width + x;

		//shared memory opitmization 
		// Load the kernel value into shared memory
		sharedKernel[threadIdx.y][threadIdx.x] = kernel[index];

		__syncthreads();  // Synchronize to ensure all threads have loaded the kernel data

		// Perform element-wise complex multiplication
		cufftComplex dataValue = data[index];
		cufftComplex kernelValue = sharedKernel[threadIdx.y][threadIdx.x];
		cufftComplex result;

		result.x = dataValue.x * kernelValue.x - dataValue.y * kernelValue.y;
		result.y = dataValue.x * kernelValue.y + dataValue.y * kernelValue.x;

		// Store the result back to the data array
		data[index] = result;
	}
}


// Function to create a 2D Gaussian kernel
cv::Mat createGaussianKernel(int size, double sigma) {
	cv::Mat kernel(size, size, CV_64F);
	double sum = 0.0;
	int center = size / 2;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			int x = i - center;
			int y = j - center;
			kernel.at<double>(i, j) = exp(-(x * x + y * y) / (2.0 * sigma * sigma)) / (2.0 * M_PI * sigma * sigma);
			sum += kernel.at<double>(i, j);
		}
	}

	// Normalize the kernel
	kernel /= sum;

	return kernel;
}

// Function to shift quadrants of an image
void shiftQuadrants(cv::Mat& image) {
	int cx = image.cols / 2;
	int cy = image.rows / 2;

	cv::Mat q0(image, cv::Rect(0, 0, cx, cy)); // Top-Left
	cv::Mat q1(image, cv::Rect(cx, 0, cx, cy)); // Top-Right
	cv::Mat q2(image, cv::Rect(0, cy, cx, cy)); // Bottom-Left
	cv::Mat q3(image, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}


// Function to apply Gaussian blur using FFT
cv::Mat applyGaussianBlurFFT(const cv::Mat& inputImage, int kernelSize, double sigma) {
	cv::Mat kernel = createGaussianKernel(kernelSize, sigma);
	cv::Mat paddedKernel(inputImage.rows, inputImage.cols, CV_64FC1, cv::Scalar(0));

	// Copy the kernel to the center of the paddedKernel
	int dx = (paddedKernel.cols - kernel.cols) / 2;
	int dy = (paddedKernel.rows - kernel.rows) / 2;
	kernel.copyTo(paddedKernel(cv::Rect(dx, dy, kernel.cols, kernel.rows)));

	// Perform 2D FFT on the input image and padded kernel
	cv::Mat inputImageDouble;
	inputImage.convertTo(inputImageDouble, CV_64FC1);
	cv::Mat fftInput, fftKernel;
	cv::dft(inputImageDouble, fftInput, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(paddedKernel, fftKernel, cv::DFT_COMPLEX_OUTPUT);

	
	
	// Multiply the FFTs element-wise
	cv::Mat complexResult;
	cv::mulSpectrums(fftInput, fftKernel, complexResult, 0);

	// Perform inverse FFT to get the blurred image
	cv::Mat blurredImage;
	cv::idft(complexResult, blurredImage, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

	// Correct quadrant order by shifting
	shiftQuadrants(blurredImage);

	// Convert back to the original image data type
	blurredImage.convertTo(blurredImage, inputImage.type());

	return blurredImage;

}

int main() {
	// Declare a cv::Mat to load image and allocate memory for it
	cv::Mat inputImage = cv::imread("noise2.jpg", cv::IMREAD_GRAYSCALE);
	if (inputImage.empty()) {
		std::cerr << "Could not open or find the image!" << std::endl;
		return -1;
	}

	// Convert the grayscale image to 64-bit floating-point
	cv::Mat inputImageDouble;
	inputImage.convertTo(inputImageDouble, CV_64FC1);


	// cpu timer started
	double cpuStart = static_cast<double>(cv::getTickCount());

	// Apply Gaussian blur with kernel size 9x9 and sigma 2
	int kernelSize = 12;
	double sigma = 2;
	cv::Mat blurredImage = applyGaussianBlurFFT(inputImage, kernelSize, sigma);
	
	//cpu timer ended
	double cpuEnd = static_cast<double>(cv::getTickCount());
	
	//calculate CPU time
	double cpuTime = (cpuEnd - cpuStart) / cv::getTickFrequency();
	std::cout << "CPU execution time: " << cpuTime << " seconds" << std::endl;
	

	cv::imwrite("output_image.jpg", blurredImage);

	//CUDA based FFT

	int width = inputImage.cols;
	int height = inputImage.rows;
	dim3 blockSize(32, 32);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	cv::Mat kernel = createGaussianKernel(12, 2);
	cv::Mat paddedKernel(height, width, CV_64FC1, cv::Scalar(0));

	// Copy the kernel to the center of the paddedKernel
	int dx = (paddedKernel.cols - kernel.cols) / 2;
	int dy = (paddedKernel.rows - kernel.rows) / 2;
	kernel.copyTo(paddedKernel(cv::Rect(dx, dy, kernel.cols, kernel.rows)));

	cufftComplex* d_data, *d_kernel;
	cudaMalloc((void**)&d_data, width * height * sizeof(cufftComplex));
	cudaMalloc((void**)&d_kernel, paddedKernel.cols * paddedKernel.rows * sizeof(cufftComplex));
	
	// Copy the image and kernel data to the GPU
	cudaMemcpy(d_data, inputImageDouble.data, width * height * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, paddedKernel.data, paddedKernel.cols * paddedKernel.rows * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	

	// Create a cuFFT plan for a 2D complex-to-complex forward FFT
	cufftHandle gpuFftInput, gpuFftKernel;
	
	cufftPlan2d(&gpuFftInput, height, width, CUFFT_C2C);
	cufftPlan2d(&gpuFftKernel, paddedKernel.rows, paddedKernel.cols, CUFFT_C2C);
	

	//initiate GPU timer
	double gpuStart = static_cast<double>(cv::getTickCount());

	// Perform the forward FFT on the GPU
	cufftExecC2C(gpuFftInput, d_data, d_data, CUFFT_FORWARD);
	cufftExecC2C(gpuFftKernel, d_kernel, d_kernel, CUFFT_FORWARD);
	

	// Perform complex element-wise multiplication with the Gaussian kernel
	kernelMultiply << <gridSize, blockSize >> > (d_data, d_kernel, width, height);
	cudaDeviceSynchronize();
	
	// Perform the inverse FFT on the GPU
	cufftExecC2C(gpuFftInput, d_data, d_data, CUFFT_INVERSE);
	
	//gpu timer ended
	double gpuEnd = static_cast<double>(cv::getTickCount());

	//calculate gpu time
	double gpuTime = (gpuEnd - gpuStart) / cv::getTickFrequency();
	std::cout << "GPU execution time: " << gpuTime << " seconds" << std::endl;

	// Copy the result back to the CPU
	cv::Mat gpuBlurredImage(height, width, CV_64FC1);
	cudaMemcpy(gpuBlurredImage.data, d_data, width * height * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	
	shiftQuadrants(gpuBlurredImage);

	blurredImage.convertTo(gpuBlurredImage, inputImage.type());
	cv::imwrite("gpu_output_image.jpg", gpuBlurredImage);
	
	// Calculate speedup
	double speedup = cpuTime / gpuTime;
	std::cout << "Speedup: " << speedup << "x" << std::endl;

	// Clean up
	cufftDestroy(gpuFftInput);
	cufftDestroy(gpuFftKernel);
	cudaFree(d_data);
	cudaFree(d_kernel);
	return 0;
}


