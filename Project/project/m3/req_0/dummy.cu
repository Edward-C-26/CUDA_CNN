#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

/*
Req 0: Using Streams to overlap computation with data transfer 
Code: memcpyasync not occuring between pinned host and device
Profiling: NOT COMPLETED
*/

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
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

    #define in_3d(i2, i1, i0) input[(i2) * (Height * Width) + (i1) * (Width) + i0]
    #define out_2d(i1, i0) output[(size_t)((i1) * (Height_out * Width_out)) + (size_t)(i0)]

    // TODO: Insert your input matrix unrolling kernel code here
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int out_width = Height_out * Width_out;

    if (t < out_width * Channel) {
        size_t c = t / out_width;    
        size_t w_unroll = t % out_width; 
                   
        size_t h_out = w_unroll / Width_out;
        size_t w_out = w_unroll % Width_out;

        int w_base = c * (K * K);             // Base offset for each channel

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                size_t h_unroll = w_base + p * K + q;
                
                // Ensure out_3d and in_4d macros align correctly with indexing
                out_2d(h_unroll, w_unroll) = in_3d(c, h_out + p, w_out + q);
            }
        }
    }
    #undef out_2d
    #undef in_3d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Height_out * Width_out;
    
    // per stream
    size_t i_size = Channel * Height * Width * sizeof(float);
    size_t o_size = Map_out * Height_out * Width_out * sizeof(float);
    size_t r_size = Height_unrolled * Width_unrolled * sizeof(float);

    // Moving from regular host to pinned host
    float *pinned_input, *pinned_output;
    cudaError_t stat_in = cudaMallocHost((void**)&pinned_input, (size_t)(i_size * Batch));
    cudaError_t stat_out = cudaMallocHost((void**)&pinned_output,(size_t)(o_size * Batch));
    cudaMemcpy(pinned_input, host_input, (size_t)(i_size * Batch), cudaMemcpyHostToHost);

    if (stat_in != cudaSuccess) {
        printf("Error allocating pinned memory");
    }
    if (stat_out != cudaSuccess) {
        printf("Error allocating pinned memory");
    }



    // Allocate device memory for mask, input, and output
    cudaMalloc((void**)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));

    // Copy mask to device (only once)
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA streams for potential parallelism
    cudaStream_t stream0; cudaStream_t stream1; cudaStream_t stream2;
    cudaStreamCreate(&stream0); cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);

    // initalize stream variables
    float *in0, *in1, *in2;
    float *out0, *out1, *out2;
    float *roll0, *roll1, *roll2; 

    // allocate memory to stream devices
    cudaMalloc((void**)&in0, i_size); cudaMalloc((void**)&in1, i_size); cudaMalloc((void**)&in2, i_size);
    cudaMalloc((void**)&out0, o_size); cudaMalloc((void**)&out1, o_size); cudaMalloc((void**)&out2, o_size);
    cudaMalloc((void**)&roll0, r_size); cudaMalloc((void**)&roll1, r_size); cudaMalloc((void**)&roll2, r_size);

    // Process batches more efficiently
    for(int b = 0; b < Batch; b += 3) {
        
        // Asynchronous input copy
        cudaMemcpyAsync(in0, (pinned_input) + b * Channel * Height * Width, i_size, cudaMemcpyHostToDevice, stream0);
        if (b + 1 < Batch) {
            cudaMemcpyAsync(in1, (pinned_input) + (b+1) * Channel * Height * Width, i_size, cudaMemcpyHostToDevice, stream1);
        }
        if (b + 2 < Batch) {
            cudaMemcpyAsync(in2, (pinned_input) + (b+2) * Channel * Height * Width, i_size, cudaMemcpyHostToDevice, stream2);
        }

        // Kernel configurations
        int num_threads = Height_out * Width_out * Channel;
        int num_blocks = ceil(1.0f * num_threads / BLOCK_SIZE);
        dim3 matmul_grid(ceil(1.0f * Width_unrolled / TILE_WIDTH), ceil(1.0f * Map_out / TILE_WIDTH), 1);
        dim3 matmul_block(TILE_WIDTH, TILE_WIDTH, 1);

        matrix_unrolling_kernel<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(in0, roll0, Channel, Height, Width, K);
        if (b + 1 < Batch) {
            matrix_unrolling_kernel<<<num_blocks, BLOCK_SIZE, 0, stream1>>>(in1, roll1, Channel, Height, Width, K);
        }
        if (b + 2 < Batch) {
            matrix_unrolling_kernel<<<num_blocks, BLOCK_SIZE, 0, stream2>>>(in2, roll2, Channel, Height, Width, K);
        }

        matrixMultiplyShared<<<matmul_grid, matmul_block, 0, stream0>>>(*device_mask_ptr, roll0, out0, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);
        if (b + 1 < Batch) {
            matrixMultiplyShared<<<matmul_grid, matmul_block, 0, stream1>>>(*device_mask_ptr, roll1, out1, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);
        }
        if (b + 2 < Batch) {
            matrixMultiplyShared<<<matmul_grid, matmul_block, 0, stream2>>>(*device_mask_ptr, roll2, out2, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);
        }

        // Asynchronous output copy
        cudaMemcpyAsync((pinned_output) + b * Map_out * Height_out * Width_out, out0, o_size, cudaMemcpyDeviceToHost, stream0);
        if (b + 1 < Batch) {
            cudaMemcpyAsync((pinned_output) + (b+1) * Map_out * Height_out * Width_out, out1, o_size, cudaMemcpyDeviceToHost, stream1);
        }
        if (b + 2 < Batch) {
            cudaMemcpyAsync((pinned_output) + (b+2) * Map_out * Height_out * Width_out, out2, o_size, cudaMemcpyDeviceToHost, stream2);
        }
    }
    // Synchronize all streams
    cudaStreamSynchronize(stream0); cudaStreamSynchronize(stream1); cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream0); cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);

    // free all device memories
    cudaFree(in0); cudaFree(in1); cudaFree(in2);
    cudaFree(out0); cudaFree(out1); cudaFree(out2);
    cudaFree(roll0); cudaFree(roll1); cudaFree(roll2);

    cudaMemcpy((void*)host_output, pinned_output, o_size * Batch, cudaMemcpyHostToHost);

    cudaFreeHost(pinned_input);
    cudaFreeHost(pinned_output);

    cudaFree(*device_input_ptr);
    cudaFree(*device_output_ptr);
    cudaFree(*device_mask_ptr);


    // Error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // all functions have been moved to conv_forward_gpu_prolog
    return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    // cudaMemcpy(host_output, device_output, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: Free device memory (This is not needed for multistream)
    // cudaFree(device_input);
    // cudaFree(device_output);
    // cudaFree(device_mask);
    return;
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