// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256
#define SCAN_BLOCK_SIZE 128

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)



//@@ insert code here
__global__ void floatToChar(unsigned char *ucharImage, float *inputImage, int imageWidth, int imageHeight, int imageChannels) {
  int pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos < imageWidth * imageHeight * imageChannels) {
    ucharImage[pos] = (unsigned char) (255 * inputImage[pos]);
  }
}

__global__ void toGrayScale(unsigned char *greyImage, unsigned char *ucharImage, int imageWidth, int imageHeight, int imageChannels) {
  int greyPos = blockDim.x * blockIdx.x + threadIdx.x;
  int colorPos = imageChannels * (blockDim.x * blockIdx.x + threadIdx.x);
  if (colorPos < imageWidth * imageHeight * imageChannels) {
    unsigned char r = ucharImage[colorPos];
    unsigned char g = ucharImage[colorPos + 1];
    unsigned char b = ucharImage[colorPos + 2];
    greyImage[greyPos] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void histGrey(unsigned int *histo, unsigned char *greyImage, int imageWidth, int imageHeight) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x; // stride = total # of threads
  // All threads in the grid collectively handle
  // blockDim.x * gridDim.x consecutive elements
  while (idx < imageHeight * imageWidth) {
    atomicAdd( &(histo[greyImage[idx]]), 1);
    idx += stride;
  }
}

// Scan Kernals
__global__ void scan_kernal_one(unsigned int *input, float *output, unsigned int *blockSums, int len, int width, int height) {
  __shared__ float T[2*BLOCK_SIZE];
  // Loading input into T
  if ((blockIdx.x * blockDim.x + threadIdx.x) * 2 >= len) {
    T[threadIdx.x * 2] = 0;
    T[threadIdx.x * 2 + 1] = 0;
  } else if ((blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1 >= len) {
    T[threadIdx.x * 2] = input[(blockIdx.x * blockDim.x + threadIdx.x) * 2] * 1.0f / width / height;
    T[threadIdx.x * 2 + 1] = 0;
  } else {
    T[threadIdx.x * 2] = input[(blockIdx.x * blockDim.x + threadIdx.x) * 2] * 1.0f / width / height;
    T[threadIdx.x * 2 + 1] = input[(blockIdx.x*blockDim.x + threadIdx.x) * 2 + 1] * 1.0f / width / height;
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
__global__ void scan_kernal_two(unsigned int *input, float *output, unsigned int *blockSums, int len) {
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

// Equalization and casting
__global__ void equalization(float *deviceOutputImageData, unsigned char *ucharImage, float *cdf, int imageWidth, int imageHeight, int imageChannels) {
  int pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos < imageWidth * imageHeight * imageChannels) {
    unsigned char correct_color = fminf(fmaxf(255.0f*(cdf[ucharImage[pos]] - cdf[0])/(1.0f - cdf[0]), 0.0f), HISTOGRAM_LENGTH - 1.0f);
    ucharImage[pos] = correct_color;
    deviceOutputImageData[pos] = (float) (ucharImage[pos]/255.0f);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char *ucharImage;
  unsigned char *greyImage;
  unsigned int *histo;
  float *cdf;
  unsigned int *blockSums;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  //@@ insert code here
  // get image data
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = (float *)malloc(imageWidth * imageHeight * imageChannels * sizeof(float));
  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&ucharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&greyImage, imageWidth * imageHeight * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&histo, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMalloc((void **)&blockSums, ceil((HISTOGRAM_LENGTH/(2*SCAN_BLOCK_SIZE))) * sizeof(unsigned int)));

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutputImageData, 0, imageWidth * imageHeight * imageChannels * sizeof(float)));
  wbCheck(cudaMemset(ucharImage, '\0', imageWidth * imageHeight * imageChannels * sizeof(unsigned char)));
  wbCheck(cudaMemset(greyImage, '\0', imageWidth * imageHeight * sizeof(unsigned char)));
  wbCheck(cudaMemset(histo, 0, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMemset(cdf, 0, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMemset(blockSums, 0, ceil(HISTOGRAM_LENGTH/(2*BLOCK_SIZE)) * sizeof(unsigned int)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice));

  // Declare dimGrid and dimBlock
  dim3 dimGridThreeChannels(ceil(imageWidth * imageHeight * imageChannels * 1.0f / BLOCK_SIZE), 1, 1);
  dim3 dimBlockThreeChannels(BLOCK_SIZE, 1, 1);

  dim3 dimGridGrey(ceil(imageWidth * imageHeight * 1.0f / BLOCK_SIZE), 1, 1);
  dim3 dimBlockGrey(BLOCK_SIZE, 1, 1);

  dim3 dimGridScanOne(ceil(HISTOGRAM_LENGTH*1.0f/(2*SCAN_BLOCK_SIZE)),1,1);
  dim3 dimBlockScanOne(SCAN_BLOCK_SIZE,1,1);
  dim3 dimGridScanTwo(ceil((HISTOGRAM_LENGTH - 2*SCAN_BLOCK_SIZE)*1.0f/SCAN_BLOCK_SIZE),1,1);
  dim3 dimBlockScanTwo(SCAN_BLOCK_SIZE,1,1);

  // Run kernals
  floatToChar<<<dimGridThreeChannels, dimBlockThreeChannels>>>(ucharImage, deviceInputImageData, imageWidth, imageHeight, imageChannels);
  toGrayScale<<<dimGridGrey, dimBlockGrey>>>(greyImage, ucharImage, imageWidth, imageHeight, imageChannels);
  histGrey<<<dimGridGrey, dimBlockGrey>>>(histo, greyImage, imageWidth, imageHeight);
  // Scan kernals
  scan_kernal_one<<<dimGridScanOne, dimBlockScanOne>>> (histo, cdf, blockSums, HISTOGRAM_LENGTH, imageWidth, imageHeight);
  scan_kernal_two<<<dimGridScanTwo, dimBlockScanTwo>>> (histo, cdf, blockSums, HISTOGRAM_LENGTH);
  // Equalization kernal
  equalization<<<dimGridThreeChannels, dimBlockThreeChannels>>>(deviceOutputImageData, ucharImage, cdf, imageWidth, imageHeight, imageChannels);

  // Copy from device into host
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost));

  // Free GPU memory
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(ucharImage);
  cudaFree(greyImage);
  cudaFree(histo);
  cudaFree(cdf);
  cudaFree(blockSums);
  wbImage_setData(outputImage, hostOutputImageData);

  wbSolution(args, outputImage);

  //@@ insert code here
  free(hostInputImageData);
  free(hostOutputImageData);
  free(inputImage);
  free(outputImage);

  return 0;
}

