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

/// Rewrites integer types of non-standard bit width, e.g. 'i48' or 'i128',
/// into types of the widths natively supported by the backend (i1, i8, i16,
/// i32, i64).
///
/// Such types are introduced by LLVM optimizations - for example SROA and
/// instcombine merge the three 'i16' components of a 'marray<short, 3>' into
/// a single 'i48' load/store - but clspv cannot lower them to valid Vulkan
/// flavoured SPIR-V.
///
/// Two different transformations are applied:
///
/// * Memory operations are *split* into several accesses of legal width, e.g.
///   a 'load i48' becomes a 'load i32' plus a 'load i16' which are recombined
///   with shift/or. Widening the access instead would read or write outside of
///   the underlying object (an 'marray<short, 3>' occupies only 6 bytes).
/// * Pure computations are *promoted* to the next larger legal width, taking
///   care to mask or sign extend operands of width-sensitive operations such
///   as 'lshr', 'ashr', 'udiv', 'sdiv' or 'icmp'.
///
/// If the pass encounters an illegal integer width it cannot rewrite, the
/// function is left unmodified and a warning is emitted, as a diagnostic is
/// more useful than the opaque failure clspv would otherwise report.
class LegalizeIntWidthsPass
    : public llvm::PassInfoMixin<LegalizeIntWidthsPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

} // namespace compiler
} // namespace hipsycl
