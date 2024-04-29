#include <cstdlib>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "gpu.hpp"
#include <chrono>
// originally designed and programmed for nvidia gpus
// by Calum Snowdon at the Australian National Univesrity 
// Copyright 2022
// Please reference Calum when using this tool and using its 
// results 


int main(int argc, char *argv[]) {
  using namespace std::chrono;

    std::cout << "              __\n";
    std::cout << "             / _)\n";
    std::cout << "    _.----._/ /\n";
    std::cout << "  /          /\n";
    std::cout << " _/ (  | (  |\n";
    std::cout << "/__.-'|_|--|_|\n";

  // Parse args
  if (argc != 5) {
    std::cout << "Need 4 arguments: m n reps reps_of_reps" << std::endl;
    exit(1);
  }

  const int m = atoi(argv[1]);
  const int n = atoi(argv[2]);
  const int reps = atoi(argv[3]);
	const int reps_of_reps = atoi(argv[4]);


  // Allocate/initialize memory and handles and such
  double *A, *w, *work;
  hipAssert(hipMalloc((void**) &A, m*n*sizeof(double)));
  hipAssert(hipMalloc((void**) &w, n*sizeof(double)));
  int* info =nullptr;
  hipAssert(hipMalloc((void**)&info, sizeof(int)));
  double *HA, *Hw;
  hipAssert(hipHostMalloc((void**)&HA, m*n*sizeof(double)));
  hipAssert(hipHostMalloc((void**) &Hw, n*sizeof(double)));

	srand(time(NULL));
  for (int i=0; i<m*n; i++)
    HA[i] = ((double)rand() / RAND_MAX);

  hipStream_t s;
  hipAssert(hipStreamCreate(&s));

  hipsolverHandle_t handle;
  hipsolverDnCreate(&handle);
  hipsolverDnSetStream(handle, s);
  hipsolverStatus_t status;

  hipAssert(hipMemcpy(A,HA,m*n*sizeof(double),hipMemcpyHostToDevice));

  // Initial call to isolate library initialization/first call overhead
   int lwork = 0;
  status = hipsolverDsyevd_bufferSize(handle, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_LOWER,
                  m, A, n, w, &lwork);
  hipAssert(hipMalloc( (void**) &work, sizeof(double) * lwork));
  hipsolverDsyevd(handle, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_LOWER,
                  m, A, n, w, work, lwork, info);

  if (status != HIPSOLVER_STATUS_SUCCESS) {
    std::cout << "It borke" << std::endl;
  }
  hipAssert(hipDeviceSynchronize());

  // Do the computation: schedule [reps] dgemm's using the same buffers 
  // and stream, then synchronize.
  std::cout << "Performing " << reps << " repetitions of " << m << "*" << n << " " << reps_of_reps << " times " << std::endl;
std::cout << std::left << std::setw(12) << "Time (ms)" << std::setw(4) << "Rep" << std::endl;
	for(int j = 0; j < reps_of_reps; ++j){
  auto start = high_resolution_clock::now();
  for (int i=0; i<reps; i++) {
  hipsolverDsyevd(handle, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_LOWER,
                  m, A, n, w, work, lwork, info);
    }
  hipAssert(hipDeviceSynchronize());
  auto end = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(end-start);

    std::cout << std::left << std::setw(12) << duration.count() / 1000
              << std::setw(4) << j << std::endl;
}

	hipAssert(hipFree(A));
	hipAssert(hipFree(w));
	hipAssert(hipFree(work));
	hipAssert(hipHostFree(HA));
	hipAssert(hipHostFree(Hw));
  return 0;
}
