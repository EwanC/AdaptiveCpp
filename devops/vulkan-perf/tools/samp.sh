#!/bin/bash
# Poor man's sampling profiler for when perf / ptrace-attach are unavailable.
# Runs the program as a gdb child and interrupts it with SIGUSR1 to collect
# all-thread backtraces into samples.txt. Aggregate with agg.py.
#
# usage: samp.sh <ACPP_VISIBILITY_MASK> <nsamples> <prog> [args...]
# e.g.   samp.sh "omp;vk:RADV" 200 ./lat 100000000 0
#        python3 agg.py inorder_executor 50
mask=$1; ns=$2; shift 2
here=$(dirname "$(realpath "$0")")
ACPP_VISIBILITY_MASK="$mask" gdb -q -batch -x "$here/samp.gdb" --args "$@" > samples.txt 2>&1 &
sleep 4
pid=$(pgrep -n -x "$(basename "$1")")
for i in $(seq "$ns"); do kill -USR1 "$pid" 2>/dev/null; sleep 0.04; done
kill -9 "$pid"; wait
