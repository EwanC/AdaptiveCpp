#!/bin/bash
# clspv wrapper for clspv_shim.so: runs the real clspv ($CLSPV_REAL) with
# $CLSPV_EXTRA appended, so clspv flags can be tried without rebuilding
# AdaptiveCpp. Use a fresh ACPP_APPDB_DIR so kernels are re-JIT-compiled:
#   ACPP_APPDB_DIR=/tmp/appdb-mad LD_PRELOAD=$PWD/clspv_shim.so \
#   CLSPV_WRAPPER=$PWD/clspv_wrapper.sh CLSPV_REAL=/path/to/clspv \
#   CLSPV_EXTRA=--cl-mad-enable ./app
unset LD_PRELOAD CLSPV_WRAPPER # the real clspv must not be redirected again
[ -n "$CLSPV_LOG" ] && echo "clspv $CLSPV_EXTRA $*" >> "$CLSPV_LOG"
exec "${CLSPV_REAL:?set CLSPV_REAL to the clspv binary}" $CLSPV_EXTRA "$@"
