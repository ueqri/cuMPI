# FILE(GLOB_RECURSE sources "*.cu" "*.cpp" "*.c")
# SET(name "cuMPI-demo")
# CUDA_ADD_EXECUTABLE(${name} ${sources})
# add_definitions(-DCUMPI_DEBUG=2)
cuda_add_library(${PROJECT_NAME} SHARED cuMPI.cu)

target_link_libraries(${PROJECT_NAME} /usr/local/cuda/lib64/libcudart.so ${MPI_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${NCCL_LIBRARY})