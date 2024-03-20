#include <cstdlib>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <chrono>
// originally designed and programmed for nvidia gpus
// by Calum Snowdon at the Australian National Univesrity 
// Copyright 2022
// Please reference Calum when using this tool and using its 
// results 

int main(int argc, char *argv[]) {
  using namespace std::chrono;

  // Parse args
  if (argc != 7) {
    std::cout << "Need 7 arguments: m k n reps [T|N] [T|N] reps_of_reps" << std::endl;
    exit(1);
  }

  const int m = atoi(argv[1]);
  const int k = atoi(argv[2]);
  const int n = atoi(argv[3]);
  const int reps = atoi(argv[4]);
const int reps_of_reps = atoi(argv[7]);

  hipblasOperation_t op1;
  hipblasOperation_t op2;
  if (argv[5][0] == 'T') {
    op1 = HIPBLAS_OP_T;
  } else {
    op1 = HIPBLAS_OP_N;
  }
  if (argv[6][0] == 'T') {
    op2 = HIPBLAS_OP_T;
  } else {
    op2 = HIPBLAS_OP_N;
  }

  // Allocate/initialize memory and handles and such
  double *A, *B, *C;
  hipMalloc((void**) &A, m*k*sizeof(double));
  hipMalloc((void**) &B, k*n*sizeof(double));
  hipMalloc((void**) &C, m*n*sizeof(double));
  double *HA, *HB;
  hipHostMalloc((void**)&HA, m*k*sizeof(double));
  hipHostMalloc((void**) &HB, k*n*sizeof(double));
srand(time(NULL));
  for (int i=0; i<m*k; i++)
    HA[i] = ((double)rand() / RAND_MAX);
  for (int i=0; i<k*n; i++)
    HB[i] = ( (double)rand() / RAND_MAX);

  double alpha = 1.0;
  double beta = 0.0;
  hipStream_t s;
  hipStreamCreate(&s);

  hipblasHandle_t handle;
  hipblasStatus_t status;
  hipblasCreate(&handle);
  hipblasSetStream(handle,s);

  hipMemcpy(A,HA,m*k*sizeof(double),hipMemcpyHostToDevice);
  hipMemcpy(B,HB,n*k*sizeof(double),hipMemcpyHostToDevice);

  // Initial call to isolate library initialization/first call overhead
  int lda = (op1 == HIPBLAS_OP_N) ? m : k;
  int ldb = (op2 == HIPBLAS_OP_N) ? k : n;
  status = hipblasDgemm(handle, op1, op2, m, n, k,
                       &alpha, A, lda, B, ldb, &beta, C, m);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    std::cout << "It borke" << std::endl;
  }
  hipDeviceSynchronize();

  // Do the computation: schedule [reps] dgemm's using the same buffers 
  // and stream, then synchronize.
  std::cout << "Performing " << reps << " repetitions of " << m << "*" << k << " by " << k << "*" << n << std::endl;
	for(int j = 0; j < reps_of_reps; ++j){
  auto start = high_resolution_clock::now();
  for (int i=0; i<reps; i++) {
    status = hipblasDgemm(handle, op1, op2, m, n, k,
                          &alpha, A, lda, B, ldb, &beta, C, m);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "It borke" << std::endl;
    }
  }
  hipDeviceSynchronize();
  auto end = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(end-start);
  std::cout << "Time     " << duration.count() << "us" << std::endl;
  std::cout << "GFLOP/s: " << ((double)2*m*k*n*reps)/((double)duration.count())/1000 << " in iter " << j << std::endl;
}
  return 0;
}
