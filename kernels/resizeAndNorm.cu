#include "device_launch_parameters.h"
#include "device_functions.h"
#include "stdio.h"
#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 
#include <math.h>


__forceinline__ __device__ float clipp(float in, float low, float high)
{
	return (in < low) ? low : (in > high ? high : in);
}

__global__ void copyKernel(unsigned char* input, unsigned char* output, int index, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
	{
		return;
	}
	//output[i+index*size] = input[i + index * size];
	output[i + index * size * 3] = input[i];
	output[i + size + index * size * 3] = input[i + size];
	output[i + 2 * size + index * size * 3] = input[i + 2 * size];

}

__global__ void resizKernel(unsigned char *inputGpu, float *outputGpu, float* normGpu, int dstW, int dstH, int srcW, int srcH)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = i % (dstW*dstH);
	int l = i / (dstW*dstH);
	const int x = k % dstW;
	const int y = k / dstW;
	if (x >= dstW || y >= dstH)
		return;
	float ratio_h = float(srcH) / float(dstH);
	float ratio_w = float(srcW) / float(dstW);
	float x0 = float(x) * ratio_w;
	float y0 = float(y) * ratio_h;
	int left = int(clipp((float)floor(x0), 0.0f, float(srcW)));
	int top = int(clipp((float)floor(y0), 0.0f, float(srcH)));
	int right = int(clipp((float)ceil(x0), 0.0f, float(srcW)));
	int bottom = int(clipp((float)ceil(y0), 0.0f, float(srcH)));
	for (int c = 0; c < 3; ++c)
	{
		unsigned char left_top_val = inputGpu[l*srcW*srcH * 3 + top * (srcW * 3) + left * (3) + c];
		unsigned char right_top_val = inputGpu[l*srcW*srcH * 3 + top * (srcW * 3) + right * (3) + c];
		unsigned char left_bottom_val = inputGpu[l*srcW*srcH * 3 + bottom * (srcW * 3) + left * (3) + c];
		unsigned char right_bottom_val = inputGpu[l*srcW*srcH * 3 + bottom * (srcW * 3) + right * (3) + c];
		float top_lerp = left_top_val + (right_top_val - left_top_val) * (x0 - left);
		float bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * (x0 - left);
		float lerp = clipp((top_lerp + (bottom_lerp - top_lerp) * (y0 - top)), 0.0f, 255.0f);
		outputGpu[i * 3 + c] = lerp;
		//float pixelMean[3]{ 123.68, 116.779, 103.939 };
		if (c == 0)
		{
			normGpu[l*dstW*dstH * 3 + k] = float(outputGpu[i * 3 + c]) - 123.68;
		}
		if (c == 1)
		{
			normGpu[l*dstW*dstH * 3 + c * dstW*dstH + k] = float(outputGpu[i * 3 + c]) - 116.779;
		}
		if (c == 2)
		{
			normGpu[l*dstW*dstH * 3 + c * dstW*dstH + k] = float(outputGpu[i * 3 + c]) - 103.939;
		}
	}
}

extern "C" void copyImg(void* input, void* output, int index, int k)
{
	const int dim = k;
	const int BS = 512;
	const int GS = (dim + BS - 1) / BS;
	copyKernel << <GS, BS, 0>> > ((unsigned char *)input, (unsigned char *)output, index, dim);
}

extern "C" void resizeAndNorm(void* inputGpu, void* resizedOutputGpu, void* normGpu, int size, int dstW, int dstH, int srcW, int srcH)
{
	int dim = size;
	const int BS = 1024;
	const int GS = (dim + BS - 1) / BS;
	resizKernel << <GS, BS, 0 >> > ((unsigned char *)inputGpu, (float *)resizedOutputGpu, (float*)normGpu, dstW, dstH, srcW, srcH);
}