add_definitions(-DCUMPI_DEBUG=0)
cuda_add_library(${PROJECT_NAME} SHARED cuMPI.cu)

# assume CUDA toolkit installed in /usr/local/cuda/
target_link_libraries(${PROJECT_NAME} /usr/local/cuda/lib64/libcudart.so ${MPI_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${NCCL_LIBRARY})