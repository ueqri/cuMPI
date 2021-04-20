#ifndef _CUMPI_RUNTIME_H_
#define _CUMPI_RUNTIME_H_

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>

#define MPI_CHECK(cmd) do {                      \
  int e = cmd;                                   \
  if (e != MPI_SUCCESS) {                        \
    printf("Failed: MPI error %s:%d '%d'\n",     \
     __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                          \
  }                                              \
} while(0)

#define CUDA_CHECK(cmd) do {                     \
  cudaError_t e = cmd;                           \
  if (e != cudaSuccess) {                        \
    printf("Failed: CUDA error %s:%d '%s'\n",    \
     __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                          \
  }                                              \
} while(0)

#define NCCL_CHECK(cmd) do {                     \
  ncclResult_t r = cmd;                          \
  if (r != ncclSuccess) {                        \
    printf("Failed, NCCL error %s:%d '%s'\n",    \
     __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                          \
  }                                              \
} while(0)

typedef ncclComm_t cuMPI_Comm;
typedef ncclDataType_t cuMPI_Datatype;
typedef ncclRedOp_t cuMPI_Op;
typedef MPI_Status cuMPI_Status;

#define cuMPI_SUM ncclSum
#define cuMPI_PROD ncclProd
#define cuMPI_MIN ncclMin
#define cuMPI_MAX ncclMax

/**************
 * Data Types *
 **************/

// Signed 8-bits integer
#define cuMPI_INT8_T ncclInt8

// Signed 8-bits integer
#define cuMPI_CHAR ncclChar

// Unsigned 8-bits integer
#define cuMPI_UINT8_T ncclUint8

// Signed 32-bits integer
#define cuMPI_INT32_T ncclInt32

// Signed 32-bits integer
#define cuMPI_INT ncclInt

// Unsigned 32-bits integer
#define cuMPI_UINT32_T ncclUint32

// Signed 64-bits integer
#define cuMPI_INT64_T ncclInt64

// Unsigned 64-bits integer
#define cuMPI_UINT64_T ncclUint64

// [UNSUPPORTED IN MPI]
// 16-bits floating point number (half precision)
// #define ncclFloat16

// [UNSUPPORTED IN MPI]
// 16-bits floating point number (half precision)
// #define ncclHalf

// [SAME WITH MPI_FLOAT]
// 32-bits floating point number (single precision)
// #define ncclFloat32

// 32-bits floating point number (single precision)
#define cuMPI_FLOAT ncclFloat

// [SAME WITH MPI_DOUBLE]
// 64-bits floating point number (double precision)
// #define ncclFloat64

// 64-bits floating point number (double precision)
#define cuMPI_DOUBLE ncclDouble



int cuMPI_Initialized( int *flag );
int cuMPI_Init( int *argc, char ***argv );
int cuMPI_Finalize();

int cuMPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
  cuMPI_Datatype datatype, cuMPI_Op op, cuMPI_Comm comm);
int cuMPI_Sendrecv(const void *sendbuf, int sendcount, cuMPI_Datatype sendtype,
  int dest, int sendtag,
  void *recvbuf, int recvcount, cuMPI_Datatype recvtype,
  int source, int recvtag,
  cuMPI_Comm comm, cuMPI_Status *status);
int cuMPI_Bcast( void *buffer, int count, cuMPI_Datatype datatype, int root, 
  cuMPI_Comm comm );
int cuMPI_Barrier( cuMPI_Comm comm );

int cuMPI_Comm_size( cuMPI_Comm comm, int *size );
int cuMPI_Comm_rank( cuMPI_Comm comm, int *rank );


extern int myRank;                 // cuMPI comm local ranks
extern int nRanks;                 // total cuMPI comm ranks
extern int localRank;              // CUDA device ID

extern ncclUniqueId id;            // NCCL Unique ID
extern cuMPI_Comm comm;            // cuMPI comm
extern cudaStream_t defaultStream; // CUDA stream generated for each GPU
extern uint64_t hostHashs[10];     // host name hash in cuMPI
extern char hostname[1024];        // host name for identification in cuMPI

#endif
