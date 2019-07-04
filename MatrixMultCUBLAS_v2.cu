// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>



#define TYPE float

/**
 *  * Returns the current time in miliseconds.
 *   */
double getMicrotime(){
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

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(TYPE *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const TYPE *A, const TYPE *B, TYPE *C, const int m, const int k, const int n) {
	int lda=k,ldb=n,ldc=n;
	//int lda=k/*WA*/,ldb=n/*WB*/,ldc=n/*WC*/;
	const TYPE alf = 1.0f;
	const TYPE bet = 0.0f;
	const TYPE *alpha = &alf;
	const TYPE *beta = &bet;
	cudaEvent_t start, stop;
	TYPE elapsedTime;
	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
    //warmup
	////for(int i=0; i< 100;i++)
    ////    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	//double start = getMicrotime();
	//std::cout << "start timing"<< std::endl;
	cudaEventCreate(&start);
 	cudaEventRecord(start,0);
	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha,  B, ldb,A, lda, beta, C, ldc);
	

	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
 	cudaEventSynchronize(stop);

 	cudaEventElapsedTime(&elapsedTime, start,stop);
	std::cout << "elapsed time: "<< elapsedTime << " miliseconds" << std::endl;
	// Destroy the handle
	cublasDestroy(handle);
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const TYPE *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_rows_C = atoi(argv[1]);
    nr_cols_A = nr_rows_B = atoi(argv[2]);
	nr_cols_B = nr_cols_C = atoi(argv[3]);
	
	TYPE *h_A = (TYPE *)malloc(nr_rows_A * nr_cols_A * sizeof(TYPE));
	TYPE *h_B = (TYPE *)malloc(nr_rows_B * nr_cols_B * sizeof(TYPE));
	TYPE *h_C = (TYPE *)malloc(nr_rows_C * nr_cols_C * sizeof(TYPE));

	// Allocate 3 arrays on GPU
	TYPE *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(TYPE));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(TYPE));
	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(TYPE));

	// If you already have useful values in A and B you can copy them in GPU:
	// cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(TYPE),cudaMemcpyHostToDevice);
	// cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(TYPE),cudaMemcpyHostToDevice);

	// Fill the arrays A and B on GPU with random numbers
	GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
	GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

	// Optionally we can copy the data back on CPU and print the arrays
	cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(TYPE),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(TYPE),cudaMemcpyDeviceToHost);
	//std::cout << "A =" << std::endl;
	//print_matrix(h_A, nr_rows_A, nr_cols_A);
	//std::cout << "B =" << std::endl;
	//print_matrix(h_B, nr_rows_B, nr_cols_B);
   
	// Multiply A and B on GPU
	gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

	// Copy (and print) the result on host memory
	cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(TYPE),cudaMemcpyDeviceToHost);
	//std::cout << "C =" << std::endl;
	//print_matrix(h_C, nr_rows_C, nr_cols_C);

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	

	//verify results
    TYPE*C_CPU = (TYPE*) (TYPE*)malloc(sizeof(TYPE*) * nr_rows_C * nr_cols_C);
    MatrixMultiplicationCPU(h_A,h_B,C_CPU, nr_rows_A, nr_cols_A, nr_cols_B);
    int count;
    for(count=1;count<nr_rows_A * nr_cols_B;count++){
       if(abs(C_CPU[count]-h_C[count]) > 0.0001)  {printf("Not Equal, idx: %d!, C_CPU[%d]: %f, C_host[%d]: %f \n",count,count,C_CPU[count],count,h_C[count]);break;}
    }
    free(C_CPU);
  
	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}