#!/bin/bash
# The HeCBench benchmark set and arguments used for R6/R7 (M1) and R14 (M2),
# run through hecbench_run.sh on every device in $GPUDEVS and then $CPUDEVS
# (same "<gpu|cpu|gpu-cuda>|<ACPP_VISIBILITY_MASK>|<tag>" format as DEVICES).
# bfs needs graphs from gen_bfs_graph.py in $HECBENCH/src/data/bfs/.
# usage: HECBENCH=... OUT=/tmp/runs GPUDEVS="..." CPUDEVS="..." hecbench_suite.sh
here=$(dirname "$(realpath "$0")")
RUN=$here/hecbench_run.sh
export HECBENCH=${HECBENCH:-~/Development/HeCBench}
OUT=${OUT:-/tmp/hecbench-runs}
# Defaults are M2 (Intel MTL iGPU + NVIDIA Ada dGPU + Intel CPU)
GPUDEVS=${GPUDEVS:-"gpu|omp;ocl:Arc|ocl-mtl gpu|omp;vk:MTL|vk-mtl gpu-cuda|omp;cuda|cuda-ada gpu|omp;vk:NVIDIA|vk-ada"}
CPUDEVS=${CPUDEVS:-"cpu|omp;ocl:Ultra|ocl-cpu cpu|omp;vk:llvmpipe|lvp cpu|omp|omp"}
suite() {
  OUT=$1/bfs $RUN bfs ../data/bfs/graph1M_random.txt
  OUT=$1/bfsgrid $RUN bfs ../data/bfs/graph1M_grid.txt
  OUT=$1 $RUN floydwarshall 1024 100 16
  OUT=$1 $RUN gaussian -q -t -s 4096
  OUT=$1 $RUN pathfinder 100000 1000 5
  OUT=$1 $RUN memcpy 8192
  OUT=$1 $RUN black-scholes 100
  OUT=$1 $RUN binomial
  OUT=$1 $RUN mandelbrot 1000
  OUT=$1 $RUN softmax 100000 784 1 100
  OUT=$1 $RUN histogram --i=100
  OUT=$1 $RUN atomicReduction
  OUT=$1 $RUN su3 -i 100 -l 32 -t 128 -v 3 -w 1
  OUT=$1 $RUN nbody 16000 30
  # correctness triage set (BUG-5): shorter timeout, as scan re-runs its
  # failing JIT on every launch on Vulkan (BUG-7)
  TMO=180 OUT=$1 $RUN lavaMD -boxes1d 10
  TMO=180 OUT=$1 $RUN lud -s 1024 -v
  TMO=180 OUT=$1 $RUN layernorm 8 1024 12288 200
  TMO=180 OUT=$1 $RUN atomicCost 16 10
  TMO=180 OUT=$1 $RUN scan 16777216 100
  TMO=180 OUT=$1 $RUN shuffle 200000 100
  TMO=180 OUT=$1 $RUN babelstream
}
export DEVICES="$GPUDEVS"; suite "$OUT"
export DEVICES="$CPUDEVS"; suite "$OUT"
