cmake_minimum_required(VERSION 3.21)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
set (CMAKE_CXX_STANDARD 17)
project(perftool)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# GPU libraries
find_package(CUDAToolkit)

if (NOT CUDAToolkit_FOUND)
	  list(APPEND CMAKE_MODULES_PREFIX $ENV{HIP_PATH}/cmake)
list(APPEND CMAKE_PREFIX_PATH $ENV{HIP_PATH})
list(APPEND CMAKE_MODULES_PREFIX /soft/compilers/chipStar/chip-spv/20250131-release/cmake)
  find_package(HIP REQUIRED)
  #find_package(hipBLAS REQUIRED)
  #find_package(rocBLAS REQUIRED)
  #find_package(hipSOLVER REQUIRED)
  #find_package(rocSOLVER REQUIRED)
  add_definitions(-DHAVE_HIP)
    if(USE_MAGMA)
      find_package(MAGMA REQUIRED)
  add_definitions(-DHAVE_MAGMA)
  set(GPU_LIBRARIES hip::host roc::hipblas roc::rocblas roc::hipsolver roc::hipsolver magma)
  else()
  set(GPU_LIBRARIES hip::host
       -L${MATHLIB_ROOT}/lib  -L${HIPSOLVER_ROOT}/lib  -lhipblas -lhipsolver
	  # roc::hipblas roc::rocblas roc::hipsolver roc::hipsolver
	  )
  include_directories(${HIPSOLVER_ROOT}/include)
    endif()
    #enable_language(HIP)
  set_source_files_properties(
    ${SOURCES} PROPERTIES LANGUAGE HIP)
else()
  set(GPU_LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver)
  add_definitions(-DHAVE_CUDA)
endif()
message("GPU LIBS " ${GPU_LIBRARIES})


add_executable(dgemm src/dgemm.cpp)
add_executable(dsyevd src/dsyevd.cpp)
target_link_libraries(dgemm PUBLIC ${GPU_LIBRARIES})
target_link_libraries(dsyevd PUBLIC ${GPU_LIBRARIES})
