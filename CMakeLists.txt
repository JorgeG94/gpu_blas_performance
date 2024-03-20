cmake_minimum_required(VERSION 3.21)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
set (CMAKE_CXX_STANDARD 17)
project(perftool)

# GPU libraries
find_package(CUDAToolkit)

if (NOT CUDAToolkit_FOUND)
  find_package(HIP REQUIRED)
  find_package(hipBLAS REQUIRED)
  find_package(rocBLAS REQUIRED)
  set(GPU_LIBRARIES hip::host roc::rocblas roc::hipblas)
  add_definitions(-DHAVE_HIP)
else()
  set(GPU_LIBRARIES CUDA::cudart CUDA::cublas)
  add_definitions(-DHAVE_CUDA)
endif()
message("GPU LIBS " ${GPU_LIBRARIES})


add_executable(performance ${CMAKE_SOURCE_DIR}/src/performance.cpp)
target_link_libraries(performance PUBLIC ${GPU_LIBRARIES})
