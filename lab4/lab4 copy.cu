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
#define TILE_WIDTH 6;
#define kernalWidth 3;

//@@ Define constant memory for device kernel here
__constant__ float Mc[kernalWidth][kernalWidth][kernalWidth];

__global__ void conv3d(float *input, float *output, const int z_size, const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float inputTile[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;
  int tx = threadIdx.x; int ty = threadIdx.y; int tz= threadIdx.z;

  int x_o = bx * 16 + tx;
  int y_o = by * 16 + ty;
  int z_o = bz * 16 + tz;
  
  int x_i = x_o-(kernalWidth-1)/2; // MASK_WIDTH / 2
  int y_i = y_o-(kernalWidth-1)/2;  
  int z_i = z_o-(kernalWidth-1)/2;
  float Pvalue = 0;

  if ((x_i >= 0) && (x_i < x_size) && (y_i >= 0) && (y_i < y_size) && z_i >= 0 && z_i < z_size) {
    inputTile[tz][ty][tx] = input[z_i * y_size * x_size + y_i * x_sizecol_i + x_i];
  } else {
    inputTile[tz][ty][tx] = 0.0f;
  }
  __syncthreads ();

  if (ty < TILE_WIDTH && tx < TILE_WIDTH && tz < TILE_WIDTH) {
    for(int i = 0; i < kernalWidth; i++) {
      for(int j = 0; j < kernalWidth; j++) {
        for (int k = 0; k < kernalWidth; k++) {
          Pvalue += Mc[i][j][k] * inputTile[i+tz][j+ty][k+kx];
        }
      }
    }
    output[z_i][y_i][x_i] = Pvalue;
  }
  __syncthreads ();
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
  cudaMemcpyToSymbol(Mc, hostKernal, kernalLength*sizeof(float))

  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(z_size/(1.0*TILE_WIDTH - (kernalWidth-1)/2)), ceil(y_size/(1.0*TILE_WIDTH - (kernalWidth-1)/2)), ceil(x_size/(1.0*TILE_WIDTH - (kernalWidth-1)/2)));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid,dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostInput + 3, deviceInput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostOutput + 3, hostOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostKernal, hostKernal, kernalLength * sizeof(float), cudaMemcpyDeviceToHost);

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceKernal);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

