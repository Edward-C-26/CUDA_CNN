// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<len) {
    out[i] = in1[i] + in2[i];
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);

  //@@ Allocate GPU memory here
  //device memory
  int size = inputLength*sizeof(float);
  float *input1, *input2, *output;
  cudaMalloc((void **) &input1, size);
  cudaMalloc((void **) &input2, size);
  cudaMalloc((void **) &output, size);

  //@@ Copy memory to the GPU here
  cudaMemcpy(input1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(input2, hostInput2, size, cudaMemcpyHostToDevice);
  cudaMemcpy(output, hostOutput, size, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(inputLength/256.0),1,1);
  dim3 DimBlock(256,1,1);

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid, DimBlock>>>(input1, input2, output, inputLength);

  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, output, size, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(input1); cudaFree(input2); cudaFree(output);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
