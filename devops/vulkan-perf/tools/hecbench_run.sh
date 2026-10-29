#!/bin/bash
# Run one HeCBench SYCL benchmark on the four comparison devices.
# Each device gets one warm-up run (JIT) and one measured run.
#
# Build first (from the HeCBench src/<bench>-sycl directory), once per variant:
#   make CC=acpp VENDOR=AdaptiveCpp EXTRA_CFLAGS="--acpp-targets=generic" GPU=yes && cp main <bench>-gpu
#   make clean
#   make CC=acpp VENDOR=AdaptiveCpp EXTRA_CFLAGS="--acpp-targets=generic" GPU=no  && cp main <bench>-cpu
#
# usage: HECBENCH=~/HeCBench OUT=/tmp/runs DEVICES="..." hecbench_run.sh <bench> [args...]
# DEVICES entries are "<gpu|cpu>|<ACPP_VISIBILITY_MASK>|<tag>"; adjust for
# your hardware (e.g. "gpu|omp;cuda|cuda" "gpu|omp;vk:NVIDIA|nvvk").
HECBENCH=${HECBENCH:-~/Development/HeCBench}
OUT=${OUT:-/tmp/hecbench-runs}
DEVICES=${DEVICES:-"gpu|omp;hip|hip gpu|omp;vk:RADV|radv cpu|omp|omp cpu|omp;vk:llvmpipe|lvp"}
mkdir -p "$OUT"
b=$1; shift
cd "$HECBENCH/src/$b-sycl" || exit 1
for cfg in $DEVICES; do
  IFS='|' read -r t m tag <<< "$cfg"
  o=$OUT/$b.$tag
  ACPP_VISIBILITY_MASK="$m" timeout "${TMO:-600}" ./"$b-$t" "$@" > "$o.warm" 2>&1
  s=$(date +%s.%N)
  ACPP_VISIBILITY_MASK="$m" timeout "${TMO:-600}" ./"$b-$t" "$@" > "$o.log" 2>&1
  rc=$?
  echo "rc=$rc wall=$(echo "$(date +%s.%N) - $s" | bc)" >> "$o.log"
done
