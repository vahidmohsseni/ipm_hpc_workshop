//source: https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#define TYPE float
#define TILE_DIM 32

/*
 Returns the current time in miliseconds.
*/
double getMilitime(){
        struct timeval ret;
        gettimeofday(&ret, NULL);
        return ((ret.tv_sec ) * 1000000u + ret.tv_usec) / 1.e6;
}



void MatrixMultiplicationCPU(TYPE*A, TYPE*B, TYPE*C, int M, int N, int K){
  int i,j,count;
  for(i=0;i<M;++i)
      for(j=0;j<K;++j)
          for(count=0;count<N;++count){
              C[i*K+j] += A[i*N+count] * B[count*K+j];
          }
}

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

void Allocate(TYPE**A, TYPE**B, TYPE**C, int M, int N, int K){
  (*A) = (*B) = (*C) = NULL;
  (*A) = (TYPE*)malloc(sizeof(TYPE*) * M * N);
  (*B) = (TYPE*)malloc(sizeof(TYPE*) * N * K);
  (*C) = (TYPE*)malloc(sizeof(TYPE*) * M * K);

  assert((*A) != NULL);
  assert((*B) != NULL);
  assert((*C) != NULL);
}
void Fill(TYPE*A, TYPE*B, int M, int N, int K){
  int i;
  for (i=0; i<M*N; i++) A[i]= 1.0;
  for (i=0; i<N*K; i++) B[i]= 2.0;
}
void UnAllocate(TYPE**A, TYPE**B, TYPE**C){
  free((*A));
  free((*B));
  free((*C));
}

void Print2DMatrix(TYPE*A, int M, int N) {
  int i;
  for(i = 0; i < M*N; ++i){
      if((i%M)==0) printf("\n");
      printf("%f ",A[i]);
  }
  printf("\n");
}


//source: https://stackoverflow.com/questions/18815489/cuda-tiled-matrix-matrix-multiplication-with-shared-memory-and-matrix-size-whic
__global__ void MatrixMultiplication(float* A, float* B, float* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}
int main(int argc, char* argv[]){
  if(argc<4){
      printf("Input Error\n");
      return 1;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  
  TYPE*A,*B,*C, *A_Device, *B_Device, *C_Device;
  Allocate(&A,&B,&C,M,N,K);
  Fill(A,B,M,N,K);

  //Allocate in Device
  checkCuda(cudaMalloc(&A_Device, M*N*sizeof(TYPE)));
  checkCuda(cudaMalloc(&B_Device, N*K*sizeof(TYPE)));
  checkCuda(cudaMalloc(&C_Device, M*K*sizeof(TYPE)));

  //Copy to Device
  cudaMemcpy(A_Device, A, M*N*sizeof(TYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(B_Device, B, N*K*sizeof(TYPE), cudaMemcpyHostToDevice);

  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  dimGrid.x = (K + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (M + dimBlock.y - 1) / dimBlock.y;

  printf("start timing\tm=%d,n=%d,k=%d\n",M,N,K);
  double start_time = getMilitime();

  MatrixMultiplication<<<dimGrid,dimBlock>>>(A_Device,B_Device,C_Device,M,N,N,K,M,K);
  cudaDeviceSynchronize(); checkLastCudaError();

  printf("elapsed time (Tiled CUDA MatMult): %f sec\n", getMilitime()-start_time);

  //copy to Host
  cudaMemcpy(C, C_Device, M*K*sizeof(TYPE), cudaMemcpyDeviceToHost);

  //Free in Device
  checkCuda(cudaFree(A_Device));
  checkCuda(cudaFree(B_Device));
  checkCuda(cudaFree(C_Device));

  //verify results
  TYPE*C_CPU = (TYPE*) (TYPE*)malloc(sizeof(TYPE*) * M * K);
  MatrixMultiplicationCPU(A,B,C_CPU,M,N,K);
  int count;
  for(count=0;count<M * K;count++){
     if(C_CPU[count]!=C[count]) {printf("Not Equal, idx: %d!",count);break;}
  }
  free(C_CPU);
  
  //free on Host
  UnAllocate(&A,&B,&C);

  return(0);
}