#include <stdio.h>
#include <unistd.h>
#include "PreciseTimer.h"
#include "../src/cuMPI_runtime.h"

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

// test P2P SendRecv method, using cocurrent overlapping
int main() {
  cuMPI_Init(NULL, NULL);
  
  const int count = (1L << 24);
  const long long data_bytes = count * sizeof(float); // 256 MiB
  const int max_times = 64;

  float *d_send[max_times] = {}, *d_recv[max_times] = {};
  for (int i = 0; i < max_times; ++i) {
    CUDA_CHECK(cudaMalloc(&d_send[i], data_bytes));
    CUDA_CHECK(cudaMalloc(&d_recv[i], data_bytes));
  }
  
  cuMPI_Status status;
  int peer = 1 - myRank;

  cuMPI_Comm pipe[max_times];
  for (int i = 0; i < max_times; ++i) {
    cuMPI_NewGlobalComm(&pipe[i]);
  }

  toth::PreciseTimer timer;
  timer.start();

  // Added cocurrent overlapping
  for (int i = 0; i < max_times; ++i) {
    cuMPI_CocurrentStart(pipe[i]);
    cuMPI_Sendrecv(d_send[i], count, cuMPI_FLOAT, peer, 0, d_recv[i], count, cuMPI_FLOAT, localRank, 0, pipe[i], &status);
    cuMPI_CocurrentEnd(pipe[i]);
  }
  cudaDeviceSynchronize();

  timer.stop();
  double time = timer.milliseconds() / 1000.0;

  const int data_mibytes = (data_bytes >> 20);
  printf("Send & Recv NCCL tests\n");
  printf("Data Size Each Time:\t%12.6f MBytes\n", (double)data_mibytes);
  printf("Total Double Exchange:\t%12.6f GBytes\n", (double)(2 * max_times * data_mibytes / 1024));
  printf("Performed times count:\t    %d\n", max_times);
  printf("Total Time cost:\t%12.6f seconds\n", time);
  printf("Average Time cost:\t%12.6f seconds\n", time/(double)(max_times));
  printf("Average Bus width(S):\t%12.6f GBytes/s\n", (double)(max_times * data_mibytes / 1024)/time);
  printf("Average Bus width(D):\t%12.6f GBytes/s\n", (double)(2 * max_times * data_mibytes / 1024)/time);
  printf("* S: Single Direction\tD: Double Direction\n\n");

  for (int i = 0; i < max_times; ++i) {
    CUDA_CHECK(cudaFree(d_send[i]));
    CUDA_CHECK(cudaFree(d_recv[i]));
  }
  cuMPI_Finalize();
  
  return 0;
}
