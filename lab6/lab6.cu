// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 128 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void kernal_one(float *input, float *output, float *blockSums, int len) {
  __shared__ float T[2*BLOCK_SIZE];
  // Loading input into T
  if ((blockIdx.x * blockDim.x + threadIdx.x) * 2 >= len) {
    T[threadIdx.x * 2] = 0;
    T[threadIdx.x * 2 + 1] = 0;
  } else if ((blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1 >= len) {
    T[threadIdx.x * 2] = input[(blockIdx.x * blockDim.x + threadIdx.x) * 2];
    T[threadIdx.x * 2 + 1] = 0;
  } else {
    T[threadIdx.x * 2] = input[(blockIdx.x * blockDim.x + threadIdx.x) * 2];
    T[threadIdx.x * 2 + 1] = input[(blockIdx.x*blockDim.x + threadIdx.x) * 2 + 1];
  }
  __syncthreads();

  // forward pass
  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0) {
      T[index] += T[index-stride];
    }
    stride = stride*2;
  }
  
  __syncthreads();
  // backward pass 
  int count = (int) (BLOCK_SIZE*1.0f/2.0f);
  while(count != 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*count*2 - 1;
    if ((index+count) < 2*BLOCK_SIZE) {
      T[index+count] += T[index];
    }
    count = count / 2;
  }
  __syncthreads();
  // load results into output
  if ((blockIdx.x*blockDim.x + threadIdx.x) * 2 >= len) {
    // do nothing
  } else if ((blockIdx.x*blockDim.x + threadIdx.x) * 2 + 1 >= len) {
    output[(blockIdx.x*blockDim.x + threadIdx.x) * 2] = T[threadIdx.x * 2];
  } else {
    output[(blockIdx.x*blockDim.x + threadIdx.x) * 2] = T[threadIdx.x * 2];
    output[(blockIdx.x*blockDim.x + threadIdx.x) * 2 + 1] = T[threadIdx.x * 2 + 1];
  }
  __syncthreads();
  // load block sums (not using the last one)
  blockSums[blockIdx.x] = T[blockDim.x*2-1];
}

__global__ void scan(float *input, float *output, float *blockSums, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  if (blockIdx.x*blockDim.x+threadIdx.x + 2*BLOCK_SIZE < len) {
    int blocks = blockIdx.x/2;
    while (blocks >= 0) {
      output[2*blockDim.x + blockIdx.x*blockDim.x+threadIdx.x] += blockSums[blocks];
      blocks = blocks - 1;
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *blockSums;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));

  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&blockSums, ceil(numElements/(2*BLOCK_SIZE)) * sizeof(float)));

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(blockSums, 0, ceil(numElements/(2*BLOCK_SIZE)) * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimGridKernalOne(ceil(numElements*1.0f/(2*BLOCK_SIZE)),1,1);
  dim3 dimBlockKernalOne(BLOCK_SIZE,1,1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  kernal_one<<<dimGridKernalOne, dimBlockKernalOne>>> (deviceInput, deviceOutput, blockSums, numElements);

  dim3 dimGridScan(ceil((numElements - 2*BLOCK_SIZE)*1.0f/BLOCK_SIZE),1,1);
  dim3 dimBlockScan(BLOCK_SIZE,1,1);

  scan<<<dimGridScan, dimBlockScan>>> (deviceInput, deviceOutput, blockSums, numElements);


  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));

  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(blockSums);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

