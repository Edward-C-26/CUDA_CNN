#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

/*
Test batch size: 10000
Loading fashion-mnist data...Done
Loading model...Done
Conv-GPU==
Layer Time: 211.331 ms
Op Time: 50.5587 ms
Conv-GPU==
Layer Time: 152.88 ms
Op Time: 30.3197 ms

Test Accuracy: 0.8714
*/

__global__ void matrix_unrolling_kernel(const float *input, const float *weight_matrix, float *C,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K, int Map_out, int Height_unrolled, int Width_unrolled) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define out_4d(i3, i2, i1, i0) C[(size_t)(i3 * (Map_out * Height * Width)) + (size_t)((i2) * (Height_out * Width_out)) + (size_t)((i1) * (Width_out)) + (size_t)(i0)]
    
    // Matmul
    __shared__ float tileA[TILE_WIDTH * TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH * TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    int out_size = Height_out * Width_out;

    float val = 0;

    int batch = col / out_size;
    int height_fuse = (col % out_size) /  Width_out;
    int width_fuse = (col % out_size) % Width_out;

    for (int tileId = 0; tileId < (Height_unrolled - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < Map_out && tileId * TILE_WIDTH + tx < Height_unrolled) {
            tileA[ty*16 + tx] = weight_matrix[row * Height_unrolled + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty*16 + tx] = 0;
        }
        if (col < Width_unrolled && tileId * TILE_WIDTH + ty < Height_unrolled) {
            int channel = (tileId * TILE_WIDTH + ty) / K / K;
            int row_offset = (tileId * TILE_WIDTH + ty) % (K*K) / K;
            int col_offset = (tileId * TILE_WIDTH + ty) % (K*K) % K;

            tileB[ty*16 +tx] = in_4d(batch, channel, (height_fuse + row_offset), (width_fuse + col_offset));
            
        } else {
            tileB[ty*16 +tx] = 0;
        }
        __syncthreads();

        if (row < Height_unrolled && col < Width_unrolled) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty*16+i] * tileB[i*16+tx];
            }
        }
        __syncthreads();
    }

    // Permute occurs here
    if (row < Height_unrolled && col < Width_unrolled) {
        out_4d(batch, row, height_fuse, width_fuse) = val; 
    }

}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;


    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    int width_grid = Channel > Map_out ? Channel: Map_out;
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid(ceil(1.0f * Width_unrolled / TILE_WIDTH), ceil(1.0f * width_grid / TILE_WIDTH), 1);
    matrix_unrolling_kernel<<<grid, block>>>(device_input, device_mask, device_output, Batch, Channel, Height, Width, K, Map_out, Height_unrolled, Width_unrolled);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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