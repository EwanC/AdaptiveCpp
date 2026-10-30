#!/bin/bash
# Build one HeCBench SYCL benchmark in up to three variants:
#   <b>-gpu       ($ACPP, gpu_selector)
#   <b>-cpu       ($ACPP, cpu_selector)
#   <b>-gpu-cuda  ($ACPP_CUDA, gpu_selector) - only if ACPP_CUDA is set, for
#                 machines where the CUDA backend lives in a separate build
# Extra compile flags (e.g. -DBENCH_PRINT) can be passed in XFLAGS.
# usage: HECBENCH=~/HeCBench ACPP=.../acpp [ACPP_CUDA=.../acpp] hecbench_build.sh <bench>
b=$1; shift
HECBENCH=${HECBENCH:-~/Development/HeCBench}
ACPP=${ACPP:-acpp}
LOGDIR=${LOGDIR:-/tmp/hecbench-logs}
mkdir -p "$LOGDIR"
log=$LOGDIR/build.$b.log
cd "$HECBENCH/src/$b-sycl" || exit 1
: > "$log"
variants="gpu|$ACPP|yes cpu|$ACPP|no"
[ -n "$ACPP_CUDA" ] && variants="$variants gpu-cuda|$ACPP_CUDA|yes"
for v in $variants; do
  IFS='|' read -r tag cc gpu <<< "$v"
  make clean >> "$log" 2>&1
  if make -j4 CC="$cc" VENDOR=AdaptiveCpp EXTRA_CFLAGS="--acpp-targets=generic $XFLAGS" GPU=$gpu "$@" >> "$log" 2>&1; then
    mv main "$b-$tag"
    echo "$b-$tag OK"
  else
    echo "$b-$tag FAILED (see $log)"
  fi
done
make clean >> "$log" 2>&1
