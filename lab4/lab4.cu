#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 4
#define Mask_Width 3

//@@ Define constant memory for device kernel here
__constant__ float Mc[Mask_Width][Mask_Width][Mask_Width];

__global__ void conv3d(float *input, float *output, const int z_size, const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  int i = z * y_size * x_size + y * x_size + x;
  int radius = Mask_Width / 2;
  __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];


  if (x >= 0 && x < x_size && y >= 0 && y < y_size && z >= 0 && z < z_size) {
    N_ds[threadIdx.z][threadIdx.y][threadIdx.x] = input[i]; // boundary checking is missing here
  } else {
    N_ds[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();


  int curr_start_point_x = blockIdx.x * blockDim.x;
  int nxt_start_point_x = (blockIdx.x + 1) * blockDim.x;

  int curr_start_point_y = blockIdx.y * blockDim.y;
  int nxt_start_point_y = (blockIdx.y + 1) * blockDim.y;

  int curr_start_point_z = blockIdx.z * blockDim.z;
  int nxt_start_point_z = (blockIdx.z + 1) * blockDim.z;


  int Z_start_point = z - radius;
  int Y_start_point = y - radius;
  int X_start_point = x - radius;
  float Pvalue = 0;

  for (int z_it = 0; z_it < Mask_Width; z_it ++) {
    for (int y_it = 0; y_it < Mask_Width; y_it ++) {
      for (int x_it = 0; x_it < Mask_Width; x_it ++) {
        int Z_index = Z_start_point+z_it;
        int Y_index = Y_start_point+y_it;
        int X_index = X_start_point+x_it;
        if (Z_index >= 0 && Z_index < z_size && 
            Y_index >= 0 && Y_index < y_size && 
            X_index >= 0 && X_index < x_size) {
          if ((Z_index >= curr_start_point_z) && (Z_index < nxt_start_point_z) && 
              (Y_index >= curr_start_point_y) && (Y_index < nxt_start_point_y) && 
              (X_index >= curr_start_point_x) && (X_index < nxt_start_point_x)) {
            Pvalue += N_ds[threadIdx.z-radius+z_it][threadIdx.y-radius+y_it][threadIdx.x-radius+x_it] * Mc[z_it][y_it][x_it];
          } else {
            Pvalue += input[Z_index*y_size*x_size + Y_index*x_size + X_index] * Mc[z_it][y_it][x_it];
          }
        }

      }
    }
  }
  if (x >= 0 && x < x_size && y >= 0 && y < y_size && z >= 0 && z < z_size) {
    output[i] = Pvalue; // boundary checking is missing here
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;


  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first three elements were the dimensions
  cudaMalloc((void**) &deviceInput, (inputLength - 3)*sizeof(float));
  cudaMalloc((void**) &deviceOutput, (inputLength - 3)*sizeof(float));


  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3,(inputLength - 3)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength*sizeof(float));

  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size/(1.0*TILE_WIDTH)),ceil(y_size/(1.0*TILE_WIDTH)),ceil(z_size/(1.0*TILE_WIDTH)));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid,dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostInput + 3, deviceInput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

