# assume using OpenMPI
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

if [ -d "build" ]; then
  rm -rf build
fi
mkdir build
cd build
cmake ..
make -j$(procs) && \
mpirun -np 2 benchmark/cuMPI-benchmark
