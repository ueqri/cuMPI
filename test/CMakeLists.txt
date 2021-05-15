cmake_minimum_required(VERSION 2.8)

# Variables and Options
option(test "Build all tests." OFF) # TODO: integrate google-test 
set(CUDA_ARCH_CODE "sm_70" CACHE STRING "Seting by GPU compute capability")
set(CUDA_ARCH_COMPUTE "compute_70" CACHE STRING "Seting by GPU compute capability")

set (PROJECT_NAME cuMPI-test)
project(${PROJECT_NAME})

# Add custom include/ directory
# set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)
# include_directories(${COMMON_INCLUDES})

find_package(MPI REQUIRED)

find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

set(CMAKE_CXX_FLAGS "-g -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=${CUDA_ARCH_COMPUTE},code=${CUDA_ARCH_CODE};)

# Include directory
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  /usr/local/cuda/include
  ${MPI_INCLUDE_PATH}
  ${CMAKE_SOURCE_DIR}/src
)
# Link library directory
LINK_DIRECTORIES(
  /usr/lib
  /usr/local/lib
)

cuda_add_executable(
  "${PROJECT_NAME}AllReduce"
  testAllReduce.cu
)
target_link_libraries(
  "${PROJECT_NAME}AllReduce"
  ${CMAKE_BINARY_DIR}/src/libcuMPI.so
  ${NCCL_LIBRARY}
)

cuda_add_executable(
  "${PROJECT_NAME}Bcast"
  testBcast.cu
)
target_link_libraries(
  "${PROJECT_NAME}Bcast"
  ${CMAKE_BINARY_DIR}/src/libcuMPI.so
  ${NCCL_LIBRARY}
)

cuda_add_executable(
  "${PROJECT_NAME}SendRecv"
  testSendRecv.cu
)
target_link_libraries(
  "${PROJECT_NAME}SendRecv"
  ${CMAKE_BINARY_DIR}/src/libcuMPI.so
  ${NCCL_LIBRARY}
)
