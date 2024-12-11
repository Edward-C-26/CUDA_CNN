#include <cmath>
#include <iostream>
#include "../../src/layer/custom/gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

/*
Op 5: FP16 
Code: NOT Completed
Profiling: NOT COMPLETED
*/


// __global__ void matrix_unrolling_kernel(const float *input, float *output,
//                                         const int Batch, const int Channel,
//                                         const int Height, const int Width,
//                                         const int K) {
//     /*
//     Modify this function to implement the input matrix unrolling kernel.

//     Function paramter definitions:
//     input - input
//     output - output
//     Batch - batch_size (number of images in x)
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */
//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//     // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
//     // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)

//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define out_3d(i2, i1, i0) output[(size_t)((i2) * (Height_out * Batch * Width_out)) + (size_t)((i1) * (Height_out * Width_out)) + (size_t)(i0)]

//     // TODO: Insert your input matrix unrolling kernel code here
//     int t = blockIdx.x * blockDim.x + threadIdx.x;
//     int out_size = Height_out * Width_out;

//     if (t < Batch * out_size) {
//         size_t b = t / out_size;                  // Batch index
//         size_t h_out = (t / Width_out) % Height_out;
//         size_t w_out = t % Width_out;

//         for (int c = 0; c < Channel; c++) {
//             int w_base = c * (K * K);             // Base offset for each channel

//             for (int p = 0; p < K; p++) {
//                 for (int q = 0; q < K; q++) {
//                     size_t h_unroll = w_base + p * K + q;
//                     size_t w_unroll = h_out * Width_out + w_out;

//                     // Ensure out_3d and in_4d macros align correctly with indexing
//                     out_3d(h_unroll, b, w_unroll) = in_4d(b, c, h_out + p, w_out + q);
//                 }
//             }
//         }
//     }
//     #undef out_3d
//     #undef in_4d
// }

// __global__ void float_to_half(half *out, const float *in, size_t height, size_t width, size_t rounded_height, size_t rounded_width) {
//     size_t row = (blockIdx.x * blockDim.x + threadIdx.x) / rounded_width;
//     size_t col = (blockIdx.x * blockDim.x + threadIdx.x) % rounded_width;
//     if (col < rounded_width && row < rounded_height) {
//         if (col < width && row < height) {
//             out[col + row * rounded_width] = __float2half(in[col + row * width]);
//         } 
//     }
// }


// // Tiled matrix multiplication kernel. Computes C = AB
// // You don't need to modify this kernel.
// __global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
//                                      int numARows, int numAColumns,
//                                      int numBRows, int numBColumns,
//                                      int numCRows, int numCColumns)
// {
//     __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

//     int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

//     int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
//     float val = 0;

//     for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
//         if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
//             tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
//         } else {
//             tileA[ty][tx] = 0;
//         }
//         if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
//             tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
//         } else {
//             tileB[ty][tx] = 0;
//         }
//         __syncthreads();

//         if (row < numCRows && col < numCColumns) {
//             for (int i = 0; i < TILE_WIDTH; i++) {
//                 val += tileA[ty][i] * tileB[i][tx];
//             }
//         }
//         __syncthreads();
//     }

//     if (row < numCRows && col < numCColumns) {
//         C[row * numCColumns + col] = val;
//     }
// }

// // Permutes the matmul result.
// // The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// // and we need to permute it into Batch x Map_out x Height_out x Width_out.
// // You don't need to modify this kernel.
// __global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
//                                       int Batch, int image_size) {
//     int b = blockIdx.y;
//     int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
//     if (x < image_size) {
//         for (int m = 0; m < Map_out; m++) {
//             output[b * Map_out * image_size + m * image_size + x] =
//                     input[m * Batch * image_size + b * image_size + x];
//         }
//     }
// }

// __global__ void downsize_matrix(float *matmul_output, const float *out_padded_matrix, int Map_out, int Width_unrolled, int rounded_M, int rounded_N) {
//     size_t row = (blockIdx.x * blockDim.x + threadIdx.x) / rounded_N;
//     size_t col = (blockIdx.x * blockDim.x + threadIdx.x) % rounded_N;
//     if (col < Width_unrolled && row < Map_out) {
//         matmul_output[col + row * rounded_N] = __half2float(out_padded_matrix[col + row * Width_unrolled]);
//     }
// }



// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // TODO: Allocate memory and copy over the relevant data structures to the GPU
//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;

//     size_t i_size = Batch * Channel * Height * Width * sizeof(float);
//     size_t m_size = Map_out * Channel * K * K * sizeof(float);
//     size_t o_size = Batch * Map_out * Height_out * Width_out * sizeof(float);

//     cudaMalloc((void**) device_input_ptr, i_size);
//     cudaMalloc((void**) device_mask_ptr, m_size);
//     cudaMalloc((void**) device_output_ptr, o_size);

//     cudaMemcpy(*device_input_ptr, host_input, i_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_mask_ptr, host_mask, m_size, cudaMemcpyHostToDevice);

//     // Error Check
//     cudaError_t error = cudaGetLastError();
//     if(error != cudaSuccess)
//     {
//         std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//         exit(-1);
//     }
// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     const size_t Height_out = Height - K + 1;
//     const size_t Width_out = Width - K + 1;
//     const size_t Height_unrolled = Channel * K * K;
//     const size_t Width_unrolled = Batch * Height_out * Width_out;

//     float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
//     //half conversions of matrix inputs;
//     half *unrolled_matrix_half; // set size to be padded
//     half *weight_matrix_half;  // set size to be padded
//     float *out_padded_matrix;  // set size to be padded
//     size_t rounded_M = ceil(1.0f * Map_out / 16) * 16;
//     size_t rounded_K = ceil(1.0f * Height_unrolled / 16) * 16;
//     size_t rounded_N = ceil(1.0f * Width_unrolled / 16) * 16;
//     cudaMalloc((void**)&unrolled_matrix_half, (size_t) rounded_N * rounded_K * sizeof(half));
//     cudaMalloc((void**)&weight_matrix_half, (size_t)  rounded_M * rounded_K * sizeof(half));
//     cudaMalloc((void**)&out_padded_matrix, (size_t) rounded_M * rounded_N * sizeof(float));

//     float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
//     cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
//     cudaMalloc((void**)&matmul_output, Batch * Map_out * Height_out * Width_out * sizeof(float));

//     // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
//     int num_threads = Batch * Height_out * Width_out * Channel;
//     int num_blocks = ceil(1.0f * num_threads / BLOCK_SIZE);
//     matrix_unrolling_kernel<<<num_blocks, BLOCK_SIZE>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);

//     // TODO: load the float matrices into a half matrix
//     dim3 weight_grid(rounded_K/16 * rounded_M/16, 1, 1);
//     dim3 weight_block(256, 1, 1);
//     float_to_half<<<weight_grid, weight_block>>>(weight_matrix_half, device_mask, Map_out, Height_unrolled, rounded_M, rounded_K);
//     dim3 unroll_grid(rounded_N/16 * rounded_K/16, 1, 1);
//     dim3 unroll_block(256, 1, 1);
//     float_to_half<<<unroll_grid, unroll_block>>>(unrolled_matrix_half, unrolled_matrix, Height_unrolled, Width_unrolled, rounded_K, rounded_N);

//     // TODO: Set the kernel dimensions and call the matmul kernel
//     int matmul_w = ceil(1.0f * Width_unrolled / TILE_WIDTH); 
//     int matmul_h = ceil(1.0f * Map_out / TILE_WIDTH);
//     dim3 dimGrid(matmul_w, matmul_h, 1);
//     dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
//     matrixMultiplyShared<<<dimGrid, dimBlock>>>(device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);

//     // Permute the result of matrix multiplication
//     const int out_image_size = Height_out * Width_out;
//     dim3 permute_kernel_grid_dim(ceil(out_image_size / BLOCK_SIZE), Batch, 1);
//     matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
//         matmul_output, device_output, Map_out, Batch, out_image_size
//     );
    
//     cudaFree(out_padded_matrix);
//     cudaFree(matmul_output);
//     cudaFree(unrolled_matrix);
// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // TODO: Copy the output back to host
//     cudaMemcpy(host_output, device_output, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);

//     // TODO: Free device memory
//     cudaFree(device_input);
//     cudaFree(device_output);
//     cudaFree(device_mask);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }