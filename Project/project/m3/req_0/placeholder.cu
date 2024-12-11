#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define NUM_STREAMS 3

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
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

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define out_3d(i2, i1, i0) output[Batch * (i2) * (i1) + i0]
    
    // TODO: Insert your input matrix unrolling kernel code here
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int h_unroll = Height_out * Width_out;
    if (t < Batch * h_unroll) {
        size_t w_out = t % Width_out;
        size_t h_out = (t / Width_out) % Height_out;
        size_t b = t / h_unroll;
        for (int c = 0; c < Channel; c++) {
            int w_base = c * (K * K);
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    size_t w_unroll = w_base + p * K + q;
                    out_3d(h_unroll, w_unroll, t) = in_4d(b, c, h_out + p, w_out + q);
                }
            }
        }
    }
    #undef out_3d
    #undef in_4d
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
    // Calculate dimensions
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;

    // Calculate sizes
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);

    // Allocate pinned memory for input and output
    float *pinned_input, *pinned_output;
    cudaMallocHost(&pinned_input, input_size);
    cudaMallocHost(&pinned_output, output_size);
    memcpy(pinned_input, host_input, input_size);

    // Allocate device memory
    cudaMalloc((void**) device_input_ptr, input_size);
    cudaMalloc((void**) device_mask_ptr, mask_size);
    cudaMalloc((void**) device_output_ptr, output_size);

    // Copy mask data (only need to do this once)
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate base chunk size and remainder
    int base_chunk_size = Batch / NUM_STREAMS;
    int remainder = Batch % NUM_STREAMS;

    // Allocate intermediate buffers for each stream
    float *unrolled_matrices[NUM_STREAMS];
    float *matmul_outputs[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        // Calculate maximum possible chunk size for any stream
        int max_chunk = base_chunk_size + (remainder > 0 ? 1 : 0);
        cudaMalloc((void**)&unrolled_matrices[i], 
                  (size_t)max_chunk * Channel * K * K * Height_out * Width_out * sizeof(float));
        cudaMalloc((void**)&matmul_outputs[i], 
                  (size_t)max_chunk * Map_out * Height_out * Width_out * sizeof(float));
    }

    // Track current offset in the batch
    int current_offset = 0;

    // Process data in chunks
    for (int i = 0; i < NUM_STREAMS; i++) {
        // Calculate this stream's chunk size
        int chunk_size = base_chunk_size + (i < remainder ? 1 : 0);
        if (chunk_size == 0) continue; // Skip if this stream has no work

        // Calculate sizes for this chunk
        size_t chunk_input_size = chunk_size * Channel * Height * Width * sizeof(float);

        // Async memory transfer of input chunk
        cudaMemcpyAsync(*device_input_ptr + current_offset * Channel * Height * Width,
                       pinned_input + current_offset * Channel * Height * Width,
                       chunk_input_size, cudaMemcpyHostToDevice, streams[i]);

        // Update offset for next chunk
        current_offset += chunk_size;
    }

    current_offset = 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        // Calculate this stream's chunk size
        int chunk_size = base_chunk_size + (i < remainder ? 1 : 0);
        if (chunk_size == 0) continue; // Skip if this stream has no work

        // Kernel parameters for the chunk
        int chunk_threads = chunk_size * Height_out * Width_out;
        int chunk_blocks = (chunk_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Launch kernels on the stream
        matrix_unrolling_kernel<<<chunk_blocks, BLOCK_SIZE, 0, streams[i]>>>(
            *device_input_ptr + current_offset * Channel * Height * Width,
            unrolled_matrices[i],
            chunk_size, Channel, Height, Width, K
        );

        // Calculate grid dimensions for this chunk
        int chunk_Width_unrolled = chunk_size * Height_out * Width_out;
        int chunk_W_grid = (chunk_Width_unrolled + TILE_WIDTH - 1) / TILE_WIDTH;
        int H_grid = (Map_out + TILE_WIDTH - 1) / TILE_WIDTH;
        dim3 dimGrid(chunk_W_grid, H_grid, 1);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

        matrixMultiplyShared<<<dimGrid, dimBlock, 0, streams[i]>>>(
            *device_mask_ptr,
            unrolled_matrices[i],
            matmul_outputs[i],
            Map_out, Height_unrolled, Height_unrolled,
            chunk_Width_unrolled, Map_out, chunk_Width_unrolled
        );

        const int chunk_out_image_size = Height_out * Width_out;
        dim3 permute_kernel_grid_dim((chunk_out_image_size - 1) / BLOCK_SIZE + 1, chunk_size, 1);
        
        matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE, 0, streams[i]>>>(
            matmul_outputs[i],
            *device_output_ptr + current_offset * Map_out * Height_out * Width_out,
            Map_out, chunk_size, chunk_out_image_size
        );
        // Update offset for next chunk
        current_offset += chunk_size;
    }
    current_offset = 0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        // Calculate this stream's chunk size
        int chunk_size = base_chunk_size + (i < remainder ? 1 : 0);
        if (chunk_size == 0) continue; // Skip if this stream has no work

        size_t chunk_output_size = chunk_size * Map_out * Height_out * Width_out * sizeof(float);

        // Async copy back to host
        cudaMemcpyAsync(pinned_output + current_offset * Map_out * Height_out * Width_out,
                       *device_output_ptr + current_offset * Map_out * Height_out * Width_out,
                       chunk_output_size, cudaMemcpyDeviceToHost, streams[i]);

        // Update offset for next chunk
        current_offset += chunk_size;
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Copy from pinned memory to host output
    memcpy((void*)host_output, pinned_output, output_size);

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(unrolled_matrices[i]);
        cudaFree(matmul_outputs[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFreeHost(pinned_input);
    cudaFreeHost(pinned_output);

    cudaFree(*device_output_ptr);
    cudaFree(*device_input_ptr);
    cudaFree(*device_mask_ptr);

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
    return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
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