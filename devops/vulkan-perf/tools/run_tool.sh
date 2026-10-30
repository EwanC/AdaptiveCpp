#!/bin/bash
# Run a tool (built in $BIN, CUDA-toolchain builds with suffix "-cuda") on
# every comparison device: one warm-up (JIT) run, then $REPS measured runs.
# Output: $OUT/<tool>.<tag>.<rep>; summarise with tool_median.py.
# DEVS entries are "<tag>|<binary suffix>|<ACPP_VISIBILITY_MASK>"; defaults are M2.
# usage: BIN=/tmp/tools OUT=/tmp/runs run_tool.sh <tool> [args...]
BIN=${BIN:-.}
OUT=${OUT:-/tmp/vk-perf-runs}
REPS=${REPS:-3}
DEVS=${DEVS:-"ocl-mtl||omp;ocl:Arc vk-mtl||omp;vk:MTL cuda-ada|-cuda|omp;cuda vk-ada||omp;vk:NVIDIA ocl-cpu||omp;ocl:Ultra lvp||omp;vk:llvmpipe omp||omp"}
mkdir -p "$OUT"
t=$1; shift
cd "$BIN" || exit 1
for cfg in $DEVS; do
  IFS='|' read -r tag sfx m <<< "$cfg"
  ACPP_VISIBILITY_MASK="$m" timeout "${TMO:-600}" ./"$t$sfx" "$@" > "$OUT/$t.$tag.warm" 2>&1
  for r in $(seq "$REPS"); do
    ACPP_VISIBILITY_MASK="$m" timeout "${TMO:-600}" ./"$t$sfx" "$@" > "$OUT/$t.$tag.$r" 2>&1
  done
done
