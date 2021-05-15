#include "cuMPI_runtime.h"

int myRank;                     // cuMPI comm local ranks
int nRanks;                     // total cuMPI comm ranks
int localRank;                  // CUDA device ID

ncclUniqueId id;                // NCCL Unique ID
cuMPI_Comm comm;                // cuMPI comm
cudaStream_t commStream;        // CUDA stream generated for each GPU
cuMPI_Comm defaultComm;         // cuMPI comm
cudaStream_t defaultCommStream; // CUDA stream generated for each GPU
uint64_t hostHashs[10];         // host name hash in cuMPI
char hostname[1024];            // host name for identification in cuMPI
std::map<cuMPI_Comm, cudaStream_t> comm2stream;

// Test Broadcast

int testBcast() {
  cuMPI_Init(NULL, NULL);
  
  int count = 100;
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
  
  // TODO: add google-test

  printf("[Rank %d]:\n", myRank);
  for (int i = 0; i < count; ++i) {
    printf("%d ", (int)h_recv[i]);
  }
  printf("\n");

  CUDA_CHECK(cudaFree(d_send));
  CUDA_CHECK(cudaFree(d_recv));
  free(h_send);
  free(h_recv);
  cuMPI_Finalize();
  return 0;
}

int main() {
  testBcast();
  return 0;
}