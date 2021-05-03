#include "cuMPI_runtime.h"

static uint64_t getHostHash(const char *string) {
  // based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void getHostName(char *hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

int cuMPI_AllocateOneGPUPerProcess() {
  // TODO
  cuMPI_Init(NULL, NULL);
  return 0;
}

int cuMPI_Initialized(int *flag) { 
  return MPI_Initialized(flag);
}

int cuMPI_Init(int *argc, char ***argv) {

  // initializing MPI
  MPI_CHECK(MPI_Init(argc, argv));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  // calculating localRank based on hostname which is used in selecting a GPU
  // localRank -> deviceID
  // myRank    -> NCCL comm rank, myRank will bind to localRank(deviceID) in each node
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPI_CHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
                          sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p = 0; p < nRanks; p++) {
    if (p == myRank) {
      break;
    }
    if (hostHashs[p] == hostHashs[myRank]) {
      localRank++;
    }
  }

  // get NCCL **default** unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) {
    ncclGetUniqueId(&id);
  }
  MPI_CHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // picking a GPU based on localRank, allocate device buffers
  printf("Picking Device in %s: %d for MPI Rank: %d/%d\n", hostname, localRank, myRank, nRanks);
  CUDA_CHECK(cudaSetDevice(localRank));
  CUDA_CHECK(cudaStreamCreate(&commStream));
  defaultCommStream = commStream;

  printf("[%d]: from default stream %p\n", localRank, (void*)commStream);

  // initializing **default** NCCL Communicator
  NCCL_CHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  // test the initiation
  int tempRank;
  NCCL_CHECK(ncclCommUserRank(comm, &tempRank));
  assert( tempRank == myRank );
  defaultComm = comm;

  // bind comm with stream
  comm2stream[comm] = commStream;

  printf("Initiated NCCL for MPI Rank: %d/%d\n", myRank, nRanks);

  return 0;
}

int cuMPI_Finalize(){
  NCCL_CHECK(ncclCommDestroy(comm));  // finalizing NCCL
  MPI_CHECK(MPI_Finalize());          // finalizing MPI
  printf("[MPI Rank %d] Success.\n", myRank);
  return 0;
}

int cuMPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
  cuMPI_Datatype datatype, cuMPI_Op op, cuMPI_Comm comm){
  #if CUMPI_DEBUG > 1
  printf("AllReduce Communicating...\n");
  #endif
  NCCL_CHECK(ncclAllReduce((const void *)sendbuf, (void *)recvbuf, count,
                            datatype, op, comm, commStream));

  return 0;
}

int cuMPI_Sendrecv(const void *sendbuf, int sendcount, cuMPI_Datatype sendtype,
  int dest, int sendtag,
  void *recvbuf, int recvcount, cuMPI_Datatype recvtype,
  int source, int recvtag,
  cuMPI_Comm comm, cuMPI_Status *status) {
  assert(sendtag == recvtag);
  //(void*)(status), (void)(source); // variable not use

  #if CUMPI_DEBUG > 1
  printf("Send&Receive Communicating...\n");
  #endif
  // peer rank id is `dest`
  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclSend(sendbuf, sendcount, sendtype, dest, comm, commStream));
  NCCL_CHECK(ncclRecv(recvbuf, recvcount, recvtype, dest, comm, commStream));
  NCCL_CHECK(ncclGroupEnd());

  return 0;
}


int cuMPI_Complex_Sendrecv(const void *sendreal, const void *sendimag, int sendcount, cuMPI_Datatype sendtype,
  int dest, int sendtag,
  void *recvreal, void *recvimag, int recvcount, cuMPI_Datatype recvtype,
  int source, int recvtag,
  cuMPI_Comm comm, cuMPI_Status *status) {
  
  assert(sendtag == recvtag);

  #if CUMPI_DEBUG > 1
  printf("Send&Receive Complex Communicating...\n");
  #endif
  // peer rank id is `dest`
  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclSend(sendreal, sendcount, sendtype, dest, comm, commStream));
  NCCL_CHECK(ncclSend(sendimag, sendcount, sendtype, dest, comm, commStream));
  NCCL_CHECK(ncclRecv(recvreal, recvcount, recvtype, dest, comm, commStream));
  NCCL_CHECK(ncclRecv(recvimag, recvcount, recvtype, dest, comm, commStream));
  NCCL_CHECK(ncclGroupEnd());

  return 0;
}

int cuMPI_Bcast( void *buffer, int count, cuMPI_Datatype datatype, int root, 
  cuMPI_Comm comm ) {
  #if CUMPI_DEBUG > 1
  printf("Bcast Communicating...\n");
  #endif
  // Legacy in-place version of ncclBroadcast in a similar fashion to MPI_Bcast
  NCCL_CHECK(ncclBcast(buffer, count, datatype, root, comm, commStream));

  return 0;
}

int cuMPI_Barrier( cuMPI_Comm comm ) {
  // TODO
  #if CUMPI_DEBUG > 1
  printf("Barrier Waiting...\n");
  #endif
  // ncclCommGetAsyncError
  CUDA_CHECK(cudaStreamSynchronize(commStream));
  return 0;
}

int cuMPI_Comm_size(cuMPI_Comm comm, int *size) {
  NCCL_CHECK(ncclCommCount(comm, size));
  #if CUMPI_DEBUG > 1
  printf("Comm Size: [%d]\n", *size);
  #endif
  return 0;
}

int cuMPI_Comm_rank(cuMPI_Comm comm, int *rank) {
  NCCL_CHECK(ncclCommUserRank(comm, rank));
  #if CUMPI_DEBUG > 1
  printf("Comm User Rank: [%d]\n", *rank);
  #endif
  return 0;
}

int cuMPI_NewGlobalComm(cuMPI_Comm *newcomm) {
  ncclUniqueId newid;
  
  // get NCCL new unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) {
    ncclGetUniqueId(&newid);
  }
  MPI_CHECK(MPI_Bcast((void *)&newid, sizeof(newid), MPI_BYTE, 0, MPI_COMM_WORLD));

  // initializing NCCL
  NCCL_CHECK(ncclCommInitRank(newcomm, nRanks, newid, myRank));

  // test the initiation
  int tempRank;
  NCCL_CHECK(ncclCommUserRank(*newcomm, &tempRank));
  assert( tempRank == myRank );

  // create the stream, and bind it to comm
  cudaStream_t tmpStream;
  CUDA_CHECK(cudaStreamCreate(&tmpStream));
  comm2stream[*newcomm] = tmpStream;

  return 0;
}

int cuMPI_NewPipe(cuMPI_Comm *pipe) {
  // TODO
  return 0;
}

int cuMPI_CocurrentStart(cuMPI_Comm targetcomm) {
  defaultComm = comm;
  defaultCommStream = commStream;
  auto iter = comm2stream.find(targetcomm);
  assert( iter != comm2stream.end() );
  // TODO comm
  commStream = iter->second;
  return 0;
}

int cuMPI_CocurrentEnd(cuMPI_Comm targetcomm) {
  // TODO targetcomm
  comm = defaultComm;
  commStream = defaultCommStream;
  return 0;
}
