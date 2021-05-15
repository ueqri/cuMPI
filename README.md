# cuMPI: easy-to-use MPI for CUDA programming

cuMPI binds a process to a GPU device(in one node, or across multiple nodes in a network), and provides high-performance communication methods between different processes(i.e. GPU devices) based on NVIDIA NCCL. It means, every MPI process is able to utilize the compute and storage resources of its own GPU device and communicates with other MPI processes by GPU Direct Access(supported by recent architecture), as if all compute and communicating tasks move to GPU. 

Using cuMPI in GPU works, We can do almost what we can in CPU-MPI programming before. And the porting is just to add some necessary variables and replace `MPI_*` to `cuMPI_*`!

## Supported API

We support some common used MPI methods up to now, e.g. 
- **Allreduce**
- **SendRecv**
- **Broadcast**
- **Barrier**

Other APIs as follows are also well-supported:
- **MPI_Init**
- **MPI_Finalize**
- **MPI_Comm_size**
- **MPI_Comm_rank**
- **MPI_Initialized**

## Usage

### Build

```bash
git clone git@github.com:ueqri/cuMPI.git
mkdir -p cuMPI/build
cd cuMPI/build
cmake -DCUDA_ARCH_CODE="sm_70" \
      -DCUDA_ARCH_COMPUTE="compute_70" \
      ..
make -j
```

Note: Please replace `sm_70` & `compute_70` to your compute capability, lookup at https://developer.nvidia.com/cuda-gpus.

### Install

The framework will be built to shared library. So the install depends on your purpose of using cuMPI.

- If you want to use `libcuMPI.so` locally, you can copy the artifact in `/path/to/cuMPI/build/src/libcuMPI.so` to the certain directory and skip this step.

- Otherwise, you can enter the `/path/to/cuMPI/build` and run `make install` to install `cuMPI_runtime.h` in `/usr/local/include` & `libcuMPI.so` in `/usr/local/lib` by default.

### Integration

- **Step 0**: check whether the *MPI_Datatype*, *MPI_Op* and functions in old MPI codes are supported by cuMPI
- **Step 1**: change all `MPI_*` to `cuMPI_*` easily
- **Step 2**: add some configurations in your CMakeLists.txt or Makefile, referred to the CMake in benchmark or test directory.
  - find NCCL, CUDA, MPI packages
  - link libcuMPI.so to your executable
- **Step 3**: use `mpirun` or `mpiexec` and specify the right slots(explicitly the numbers of GPU devices) in single node or multiple nodes
- **Step 4**: run the all-in-GPU tasks

## Benchmark

The benchmark in `benchmark/CocurrentSendRecv.cu` uses a cocurrent communicating methods for SendRecv to test two GPUs band width.

1. If two GPUs are **in one nodes**, the benchmark shows the estimated band width of PCIe or NVLinks P2P connections. You can find the topological structure of GPUs in single node by `nvidia-smi topo -m`.

2. If two GPUs are **in different nodes**, the benchmark mainly shows the band width of InfiniBand or Ethernet connections if the NVLinks or PCIe band width is not the bottleneck.

How to run benchmark:

```bash
cd /path/to/cuMPI/build/benchmark
# Test two GPUs connection band width in single node
mpirun -np 2 ./cuMPI-benchmark
```

## Test

We provides the test for **Allreduce**, **SendRecv**, **Broadcast**.

You can run the following commands for test:

```bash
cd /path/to/cuMPI/build/test
mpirun -np 2 ./cuMPI-testBcast
mpirun -np 2 ./cuMPI-testSendRecv
mpirun -np 2 ./cuMPI-testAllReduce
```

In the next steps, we would add google-test for modernizations.
