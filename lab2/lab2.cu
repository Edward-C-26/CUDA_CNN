// LAB 2 FA24

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < numCColumns && y < numCRows) {
    float sum = 0.0;
    
    for (int i_col = 0; i_col < numAColumns; i_col++) {
      sum += A[Row*numAColumns + i_col] * B[i_col*numBColumns + Col];
    }
    C[Row*numCColumns + Col] = sum;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  int C_size = numCColumns * numCRows * sizeof(float);
  hostC = (float *)malloc(C_size);

  //@@ Allocate GPU memory here
  float *deviceA; // The A matrix
  float *deviceB; // The B matrix
  float *deviceC; // The output C matrix
  cudaMalloc((void**) &deviceA, numAColumns*numARows*sizeof(float));
  cudaMalloc((void**) &deviceB, numBColumns*numBRows*sizeof(float));
  cudaMalloc((void**) &deviceC, numCColumns*numCRows*sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA,hostA,numAColumns*numARows*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB,numBColumns*numBRows*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC,hostC,numCColumns*numCRows*sizeof(float),cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(1.0*numCColumns/16.0),ceil(1.0*numCRows/16.0),1);
  dim3 DimBlock(16,16,1);

  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid,DimBlock>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);
  cudaDeviceSynchronize();
  
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC,deviceC,numCColumns*numCRows*sizeof(float),cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  //@@Free the hostC matrix
  free(hostC);

  return 0;
}

