cmake_minimum_required(VERSION 3.21)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
set (CMAKE_CXX_STANDARD 17)
project(perftool)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# GPU libraries
find_package(CUDAToolkit)

file(GLOB SOURCES "src/*.cpp")
if (NOT CUDAToolkit_FOUND)
  find_package(HIP REQUIRED)
  find_package(hipBLAS REQUIRED)
  find_package(rocBLAS REQUIRED)
  add_definitions(-DHAVE_HIP)
    if(USE_MAGMA)
      find_package(MAGMA REQUIRED)
  add_definitions(-DHAVE_MAGMA)
  set(GPU_LIBRARIES hip::host roc::rocblas roc::hipblas magma)
		else()
  set(GPU_LIBRARIES hip::host roc::rocblas roc::hipblas)
    endif()
enable_language(HIP)
  set_source_files_properties(
    ${SOURCES} PROPERTIES LANGUAGE HIP)
else()
  set(GPU_LIBRARIES CUDA::cudart CUDA::cublas)
  add_definitions(-DHAVE_CUDA)
endif()
message("GPU LIBS " ${GPU_LIBRARIES})


add_executable(performance ${SOURCES})
target_link_libraries(performance PUBLIC ${GPU_LIBRARIES})
