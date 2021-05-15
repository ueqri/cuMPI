add_definitions(-DCUMPI_DEBUG=0)
cuda_add_library(${PROJECT_NAME} SHARED cuMPI.cu)

# assume CUDA toolkit installed in /usr/local/cuda/
target_link_libraries(${PROJECT_NAME} /usr/local/cuda/lib64/libcudart.so ${MPI_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${NCCL_LIBRARY})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER cuMPI_runtime.h)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include)