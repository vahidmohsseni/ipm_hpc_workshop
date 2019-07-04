#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>

/*
 Returns the current time in miliseconds.
*/
double getMilitime(){
        struct timeval ret;
        gettimeofday(&ret, NULL);
        return ((ret.tv_sec ) * 1000000u + ret.tv_usec) / 1.e6;
}




#define TYPE double


#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

__global__ void ReductionKernel(TYPE* ArrDevice, TYPE* ArrTempDevice, long ArrSize){
  unsigned long objectId = blockDim.x * blockIdx.x + threadIdx.x;
  if (objectId < ArrSize) {
      atomicAdd(&ArrTempDevice[0], ArrDevice[objectId]);
  }
}

__global__ void PrintKernel(TYPE* ArrDevice){
   printf("ArrSumDevice: %f \n",ArrDevice[0]);
}


int main(int argc, char* argv[]){
  if(argc<2){
      printf("Input Error\n");
      return 1;
  }

  long ArrSize = atoi(argv[1]);
  TYPE *ArrDevice, *ArrTempDevice,*ArrHost=NULL;

  //Allocate in host
  ArrHost = (TYPE*) malloc(ArrSize*sizeof(TYPE));
  assert(ArrHost != NULL) ;

  //Fill in host
  long count;
  for(count=0;count<ArrSize;count++){
      ArrHost[count] = 1;
  }

  //Allocate in device
  checkCuda(cudaMalloc(&ArrDevice, ArrSize*sizeof(TYPE)));
  checkCuda(cudaMalloc(&ArrTempDevice, ArrSize*sizeof(TYPE)));

  //Fill in device
  cudaMemcpy(ArrDevice, ArrHost, ArrSize*sizeof(TYPE), cudaMemcpyHostToDevice);
  PrintKernel<<< 1, 1 >>>(ArrDevice);
  cudaDeviceSynchronize(); checkLastCudaError();
  
  //Call kernel
  unsigned long numThreadsPerClusterBlock = 512;
  unsigned long numClusterBlocks =
        (ArrSize + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf("start timing\n");

  ////printf("numThreadsPerClusterBlock: %d \n",numThreadsPerClusterBlock);
  ////printf("numClusterBlocks: %d \n",numClusterBlocks);

  checkCuda( cudaEventRecord(start, 0) );
  
  double start_time = getMilitime();
  ReductionKernel<<< numClusterBlocks, numThreadsPerClusterBlock>>>
          (ArrDevice, ArrTempDevice,ArrSize);

  ////cudaDeviceSynchronize(); checkLastCudaError();

  printf("elapsed time (CPU): %f sec\n", getMilitime()-start_time);

  checkCuda( cudaEventRecord(stop, 0) );
  checkCuda( cudaEventSynchronize(stop) );


  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("elapsed time (CUDA): %f milisec\n", milliseconds);

  //Copy result to host (maybe prlong in the device)
  PrintKernel<<< 1, 1 >>>(ArrTempDevice);
  cudaDeviceSynchronize(); checkLastCudaError();

  //free device data
  checkCuda(cudaFree(ArrDevice));
  checkCuda(cudaFree(ArrTempDevice));

  free(ArrHost);
  return(0);
}