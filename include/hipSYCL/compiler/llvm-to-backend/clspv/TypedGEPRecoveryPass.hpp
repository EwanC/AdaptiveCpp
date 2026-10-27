/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#pragma once

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

// Rewrites byte-granularity GEPs with a non-constant offset,
//   getelementptr i8, ptr %p, i64 %off
// into a typed GEP,
//   getelementptr T, ptr %p, i64 (ashr exact %off, log2(sizeof(T)))
// when every memory access reachable through the GEP (looking through
// GEP/PHI/select chains) uses the same power-of-two sized type T and %off is
// known to be a multiple of sizeof(T).
//
// InstCombine can fold the scaling of a run-time index into the offset and
// emit a byte GEP instead, e.g. for `p += n` on an int pointer with a 32-bit n:
//   getelementptr i8, ptr %p, i64 (ashr exact (shl i64 %n, 32), 30)
// Such GEPs are common at ACPP_ADAPTIVITY_LEVEL=0, where work-group sizes are
// run-time values. clspv's SimplifyPointerBitcastPass can loop forever on
// pointer induction variables advanced this way. Complements
// FoldChainedGEPsPass, which handles constant byte offsets. Must run after the
// last InstCombine invocation, which would undo the rewrite.
class TypedGEPRecoveryPass : public llvm::PassInfoMixin<TypedGEPRecoveryPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

} // namespace compiler
} // namespace hipsycl
