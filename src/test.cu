#include "cuMPI_runtime.h"

int testBcast() {
  cuMPI_Init(NULL, NULL);
  
  int count = 50;
  float *h_send = (float *)malloc(count * sizeof(float)),
        *h_recv = (float *)malloc(count * sizeof(float));
  if (myRank == 0) {
    for (int i = 0; i < count; ++i) {
      h_send[i] = 2 * i + myRank;
    }
  }

  float *d_send = NULL, *d_recv = NULL;
  CUDA_CHECK(cudaMalloc(&d_send, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_recv, count * sizeof(float)));
  
  CUDA_CHECK(cudaMemcpy(d_send, h_send, count * sizeof(float), cudaMemcpyHostToDevice));

  cuMPI_Bcast(d_send, count, cuMPI_FLOAT, 0, comm);

  CUDA_CHECK(cudaMemcpy(h_recv, d_send, count * sizeof(float), cudaMemcpyDeviceToHost));
  
  printf("[%d]:\n", myRank);
  for (int i = 0; i < count; ++i) {
    printf("%d ", (int)h_recv[i]);
  }

  CUDA_CHECK(cudaFree(d_send));
  CUDA_CHECK(cudaFree(d_recv));
  free(h_send);
  free(h_recv);
  cuMPI_Finalize();
  return 0;
}

int testSendRecv() {
  cuMPI_Init(NULL, NULL);
  
  int count = 50;
  float *h_send = (float *)malloc(count * sizeof(float)),
        *h_recv = (float *)malloc(count * sizeof(float));
  for (int i = 0; i < count; ++i) {
    h_send[i] = 2 * i + localRank;
  }

  float *d_send = NULL, *d_recv = NULL;
  CUDA_CHECK(cudaMalloc(&d_send, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_recv, count * sizeof(float)));
  
  CUDA_CHECK(cudaMemcpy(d_send, h_send, count * sizeof(float), cudaMemcpyHostToDevice));

  cuMPI_Status status;
  int peer = 1 - myRank;
  cuMPI_Sendrecv(d_send, count, cuMPI_FLOAT, peer, 0, d_recv, count, cuMPI_FLOAT, localRank, 0, comm, &status);

  CUDA_CHECK(cudaMemcpy(h_recv, d_recv, count * sizeof(float), cudaMemcpyDeviceToHost));
  
  printf("[%d]:\n", myRank);
  for (int i = 0; i < count; ++i) {
    printf("%d->%d ", (int)h_send[i], (int)h_recv[i]);
  }

  CUDA_CHECK(cudaFree(d_send));
  CUDA_CHECK(cudaFree(d_recv));
  free(h_send);
  free(h_recv);
  cuMPI_Finalize();
  return 0;
}

int testAllReduce() {
  cuMPI_Init(NULL, NULL);
  
  int count = 50;
  float *h_send = (float *)malloc(count * sizeof(float)),
        *h_recv = (float *)malloc(count * sizeof(float));
  for (int i = 0; i < count; ++i) {
    h_send[i] = 2 * i + myRank;
  }

  float *d_send = NULL, *d_recv = NULL;
  CUDA_CHECK(cudaMalloc(&d_send, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_recv, count * sizeof(float)));
  
  CUDA_CHECK(cudaMemcpy(d_send, h_send, count * sizeof(float), cudaMemcpyHostToDevice));

  cuMPI_Allreduce(d_send, d_recv, count, cuMPI_FLOAT, cuMPI_SUM, comm);

  CUDA_CHECK(cudaMemcpy(h_recv, d_recv, count * sizeof(float), cudaMemcpyDeviceToHost));
  
  printf("[%d]:\n", myRank);
  for (int i = 0; i < count; ++i) {
    printf("%d->%d ", (int)h_send[i], (int)h_recv[i]);
  }

  CUDA_CHECK(cudaFree(d_send));
  CUDA_CHECK(cudaFree(d_recv));
  free(h_send);
  free(h_recv);
  cuMPI_Finalize();
  return 0;
}
