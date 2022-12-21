#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <chrono>
// originally designed and programmed for nvidia gpus
// by Calum Snowdon at the Australian National Univesrity 
// Copyright 2022
// Please reference Calum when using this tool and using its 
// results 
int main(int argc, char *argv[]) {
  using namespace std::chrono;
  if (argc != 7) {
    std::cout << "Need 4 arguments: m k n reps" << std::endl;
    exit(1);
  }
  const int m = atoi(argv[1]);
  const int k = atoi(argv[2]);
  const int n = atoi(argv[3]);
  const int reps = atoi(argv[4]);
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
  double *A, *B, *C;
  hipMalloc((void**) &A, m*k*sizeof(double));
  hipMalloc((void**) &B, k*n*sizeof(double));
  hipMalloc((void**) &C, m*n*sizeof(double));
  double *A2, *B2, *C2;
  hipMalloc((void**) &A2, m*k*sizeof(double));
  hipMalloc((void**) &B2, k*n*sizeof(double));
  hipMalloc((void**) &C2, m*n*sizeof(double));
  double *HA, *HB, *HC;
  double *HA2, *HB2, *HC2;
  hipHostMalloc((void**)&HA, m*k*sizeof(double));
  hipHostMalloc((void**) &HB, k*n*sizeof(double));
  hipHostMalloc((void**) &HC, m*n*sizeof(double));
  hipHostMalloc((void**)&HA2, m*k*sizeof(double));
  hipHostMalloc((void**) &HB2, k*n*sizeof(double));
  hipHostMalloc((void**) &HC2, m*n*sizeof(double));
  for (int i=0; i<m*k; i++)
    HA[i] = ((double) i);
  for (int i=0; i<k*n; i++)
    HB[i] = ((double) i);
  double alpha = 1.0;
  double beta = 0.0;
  hipStream_t s1;
  hipStream_t s2;
  hipStreamCreateWithFlags(&s1, hipStreamNonBlocking);
  hipStreamCreateWithFlags(&s2, hipStreamNonBlocking);
  hipblasHandle_t handle1;
  hipblasHandle_t handle2;
  hipblasCreate(&handle1);
  hipblasCreate(&handle2);
  hipblasSetStream(handle1,s1);
  hipblasSetStream(handle2,s2);
  hipblasStatus_t status;
  std::cout << "Performing " << reps << " repetitions of " << m << " " << k << " " << n << std::endl;
  hipMemcpy(A,HA,m*k*sizeof(double),hipMemcpyHostToDevice);
  hipMemcpy(B,HB,n*k*sizeof(double),hipMemcpyHostToDevice);
  hipMemcpy(A2,HA2,m*k*sizeof(double),hipMemcpyHostToDevice);
  hipMemcpy(B2,HB2,n*k*sizeof(double),hipMemcpyHostToDevice);
  int lda = (op1 == HIPBLAS_OP_N) ? m : k;
  int ldb = (op2 == HIPBLAS_OP_N) ? k : n;
  status = hipblasDgemm(handle1, op1, op2, m, n, k,
                       &alpha, A, lda, B, ldb, &beta, C, m);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    std::cout << "It borke" << std::endl;
  }
  //std::cout << "A" << std::endl;
  //for (int i=0; i<k; i++) {
  //  for (int j=0; j<m; j++) {
  //    std::cout << HA[j*k+i] << " ";
  //  }
  //  std::cout << std::endl;
  //}
  //std::cout << "B" << std::endl;
  //for (int i=0; i<k; i++) {
  //  for (int j=0; j<n; j++) {
  //    std::cout << HB[i*n+j] << " ";
  //  }
  //  std::cout << std::endl;
  //}
  hipDeviceSynchronize();

  auto start = high_resolution_clock::now();
  for (int i=0; i<reps; i++) {
    double **Am = &A;
    double **Bm = &B;
    // A2 = A^T
    //std::cout << "A" << std::endl;
    //for (int i=0; i<k; i++) {
    //  for (int j=0; j<m; j++) {
    //    std::cout << HA[j*k+i] << " ";
    //  }
    //  std::cout << std::endl;
    //}
    //std::cout << "A^T" << std::endl;
    //for (int i=0; i<m; i++) {
    //  for (int j=0; j<k; j++) {
    //    std::cout << HA2[j*m+i] << " ";
    //  }
    //  std::cout << std::endl;
    //}
    // C = AB
    //status = hipblasDgemm(handle1, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k,
    //                     &alpha, A, m, B, k, &beta, C, m);
    // C2 = A^TB
    //if (op1 == HIPBLAS_OP_T) {
    //  status = hipblasDgeam(handle1, HIPBLAS_OP_T,  HIPBLAS_OP_N,
    //                      m, k, &alpha,
    //                      A, k, &beta,
    //                      A2, m,
    //                      A2, m);
    //  Am = &A2;
    //}
    //if (op2 == HIPBLAS_OP_N) {
    //  status = hipblasDgeam(handle1, HIPBLAS_OP_T,  HIPBLAS_OP_N,
    //                      k, n, &alpha,
    //                      B, n, &beta,
    //                      B2, k,
    //                      B2, k);
    //  Bm = &B2;
    //}
    status = hipblasDgemm(handle1, op1, op2, m, n, k,
                         &alpha, *Am, lda, *Bm, ldb, &beta, C2, m);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "It borke" << std::endl;
    }
  }
  hipDeviceSynchronize();
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end-start);
  std::cout << "Time     " << duration.count() << "us" << std::endl;
  std::cout << "GFLOP/s: " << ((double)2*m*k*n*reps)/((double)duration.count())/1000 << std::endl;;
  //std::cout << "C" << std::endl;
  //for (int i=0; i<m; i++) {
  //  for (int j=0; j<n; j++) {
  //    std::cout << HC[i*n+j] << " ";
  //  }
  //  std::cout << std::endl;
  //}
  //std::cout << "C2" << std::endl;
  //for (int i=0; i<m; i++) {
  //  for (int j=0; j<n; j++) {
  //    std::cout << HC2[i*n+j] << " ";
  //  }
  //  std::cout << std::endl;
  //}
}
