
#ifdef HAVE_HIP
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#elif HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define hipblasOperation_t cublasOperation_t
#define HIPBLAS_OP_T CUBLAS_OP_T
#define HIPBLAS_OP_N CUBLAS_OP_N
#define hipMalloc cudaMalloc
#define hipHostMalloc cudaMallocHost
#define hipStream_t cudaStream_t
#define hipStreamCreate cudaStreamCreate
#define hipblasHandle_t cublasHandle_t
#define hipblasStatus_t cublasStatus_t 
#define hipblasCreate cublasCreate
#define hipblasSetStream cublasSetStream
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipblasDgemm cublasDgemm
#define HIPBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#endif 

#define hipAssert(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true) {
    if (code != hipSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#ifdef HAVE_MAGMA
#include <magma_v2.h>
#define hipblasOperation_t magma_trans_t
#define HIPBLAS_OP_T MagmaTrans
#define HIPBLAS_OP_N MagmaNoTrans
#endif
