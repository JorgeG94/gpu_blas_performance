cmake_minimum_required(VERSION 3.21)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
set (CMAKE_CXX_STANDARD 17)
project(perftool)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# GPU libraries
find_package(CUDAToolkit)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
# nlohmann_json
if(EXESS_USE_EXTERNAL_JSON)
  find_package(nlohmann_json 3.11.3 REQUIRED)
else()
  include(FetchContent)
  FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/json")
  FetchContent_MakeAvailable(json)
endif()


if (NOT CUDAToolkit_FOUND)
  find_package(HIP REQUIRED)
  find_package(hipBLAS REQUIRED)
  find_package(rocBLAS REQUIRED)
  find_package(hipSOLVER REQUIRED)
  find_package(rocSOLVER REQUIRED)
  add_definitions(-DHAVE_HIP)
    if(USE_MAGMA)
      find_package(MAGMA REQUIRED)
  add_definitions(-DHAVE_MAGMA)
  set(GPU_LIBRARIES hip::host roc::hipblas roc::rocblas roc::hipsolver roc::hipsolver magma)
		else()
  set(GPU_LIBRARIES hip::host roc::hipblas roc::rocblas roc::hipsolver roc::hipsolver)
    endif()
enable_language(HIP)
  set_source_files_properties(
    ${SOURCES} PROPERTIES LANGUAGE HIP)
else()
  set(GPU_LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver)
  add_definitions(-DHAVE_CUDA)
endif()
message("GPU LIBS " ${GPU_LIBRARIES})


add_executable(dgemm src/dgemm.cpp)
add_executable(dsyevd src/dsyevd.cpp)
target_link_libraries(dgemm PUBLIC ${GPU_LIBRARIES}
   MPI::MPI_CXX
         OpenMP::OpenMP_CXX
         hdf5::hdf5_cpp
         hdf5::hdf5
         nlohmann_json::nlohmann_json
         LAPACK::LAPACK
         BLAS::BLAS
         -lstdc++fs
         )
target_link_libraries(dsyevd PUBLIC ${GPU_LIBRARIES})
