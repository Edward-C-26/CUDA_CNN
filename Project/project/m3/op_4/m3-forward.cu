#include <cmath>
#include <iostream>
#include <cublas_v2.h>
#include "../../src/layer/custom/gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

/*
op 4: Using cuBLAS for matrix multiplication
code: COMPELTED
profiling: NOT COMPELTED

Test batch size: 100
Loading fashion-mnist data...Done
Loading model...Done
Conv-GPU==
Layer Time: 24.8001 ms
Op Time: 20.6273 ms
Conv-GPU==
Layer Time: 4.36212 ms
Op Time: 1.23862 ms

Test Accuracy: 0.86

*/

__global__ void matrix_unrolling_kernel(const float*  input, float*  output,
                                        const int Batch, const int  Channel,
                                        const int  Height, const int  Width,
                                        const int  K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define out_3d(i2, i1, i0) output[(size_t)((i2) * (Height_out * Batch * Width_out)) + (size_t)((i1) * (Height_out * Width_out)) + (size_t)(i0)]

    // TODO: Insert your input matrix unrolling kernel code here
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int out_size = Height_out * Width_out;

    if (t < Batch * out_size) {
        size_t b = t / out_size;                  // Batch index
        size_t h_out = (t / Width_out) % Height_out;
        size_t w_out = t % Width_out;

        #pragma unroll(4) 
        for (int c = 0; c < Channel; c++) {
            int w_base = c * (K * K);             // Base offset for each channel

            #pragma unroll(8)
            for (int p = 0; p < K; p++) {
                #pragma unroll(8)
                for (int q = 0; q < K; q++) {
                    size_t h_unroll = w_base + p * K + q;
                    size_t w_unroll = h_out * Width_out + w_out;

                    // Ensure out_3d and in_4d macros align correctly with indexing
                    out_3d(h_unroll, b, w_unroll) = in_4d(b, c, h_out + p, w_out + q);
                }
            }
        }
    }
    #undef out_3d
    #undef in_4d
}


// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float*  input, float*  output, int  Map_out,
                                      int  Batch, int  image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        #pragma unroll(4)
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float*  host_output, const float*  host_input, const float*  host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    size_t i_size = Batch * Channel * Height * Width * sizeof(float);
    size_t m_size = Map_out * Channel * K * K * sizeof(float);
    size_t o_size = Batch * Map_out * Height_out * Width_out * sizeof(float);

    cudaMalloc((void**) device_input_ptr, i_size);
    cudaMalloc((void**) device_mask_ptr, m_size);
    cudaMalloc((void**) device_output_ptr, o_size);

    cudaMemcpy(*device_input_ptr, host_input, i_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, m_size, cudaMemcpyHostToDevice);

    // Error Check
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float*  device_output, const float*  device_input, const float*  device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;

    float *unrolled_matrix;
    float *matmul_output;
    cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)&matmul_output, (size_t) (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    int num_threads = Batch * Height_out * Width_out;
    int num_blocks = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matrix_unrolling_kernel<<<num_blocks, BLOCK_SIZE>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,Width_unrolled,Map_out,Height_unrolled,&alpha,unrolled_matrix,Width_unrolled,device_mask,Height_unrolled,&beta,matmul_output,Width_unrolled);

    cublasDestroy(handle);

    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float*  device_input, float*  device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}