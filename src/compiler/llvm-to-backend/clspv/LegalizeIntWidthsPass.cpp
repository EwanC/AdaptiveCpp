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

// This pass legalizes integer types of non-standard bit width before the IR is
// handed to clspv.
//
// Background
// ----------
// SSCP compiling e.g. `sycl::marray<short, 3>` results in LLVM optimizations
// (SROA/instcombine) merging the three `i16` elements into a single `i48`
// value. Similar types appear for other data types, e.g. `i128` is created on
// aarch64 hosts for `sycl::vec<float, 3>`. clspv cannot lower such types to
// Vulkan flavoured SPIR-V, which only supports 8/16/32/64-bit integers.
//
// Two distinct transformations are required:
//
// * Memory accesses are *split* into accesses of legal width. A `load i48`
//   reads 6 bytes, so widening it to a `load i64` would read beyond the
//   object; it is instead replaced by a `load i32` and a `load i16` that are
//   recombined via shift/or. The same applies in reverse for stores.
// * Computations are *promoted* to the next larger legal width. Promoted
//   values only carry meaningful information in their low `N` bits, so
//   operands of width-sensitive operations are masked (unsigned) or sign
//   extended (signed) before use.
//
// Anything that cannot be handled leaves the function untouched and emits a
// warning, which is a better diagnostic than the opaque clspv failure.
//
// Traversal algorithm
// -------------------
// The rewrite is driven by a single reverse post-order (RPO) walk of the
// function instead of an iterative worklist fixpoint. The pass runs in six
// phases, see `IntWidthLegalizer::run()`:
//
//   0. `splitWideStores()`   - handle values too wide to promote (>64 bits)
//   1. signature check       - bail out on illegal argument/return types
//   2. `collectWork()`       - RPO scan, legality check, build work list
//   3. `rewrite()`           - rewrite in RPO order, patch phis afterwards
//   4. erase originals       - two-step teardown to break phi cycles
//   5. alloca rewrite        - illegal alloca types become byte arrays
//   6. `warnOnRemainingIllegalTypes()` - diagnostic sweep
//
// Why reverse post-order
// ~~~~~~~~~~~~~~~~~~~~~~
// RPO visits a basic block only after all of its predecessors, except across
// loop back edges. Combined with the natural top-to-bottom order inside a
// basic block, this means that for every instruction `I` processed in phase 3,
// every operand of `I` that is itself an instruction has *already* been
// rewritten and therefore has an entry in the `Promoted` map. The only
// exception is a phi operand arriving on a back edge, which is why phi nodes
// get a placeholder and are patched in a second pass over `PhiMap`.
//
// A single ordered pass is possible because the rewrite is a pure value
// mapping: `Promoted[V]` holds a value of legal type whose low `N` bits equal
// the low `N` bits of the original `N`-bit value `V` (bits above `N` are
// undefined). Nothing about the mapping of an instruction depends on its
// users, so no information flows backwards and no fixpoint iteration is
// needed. This keeps the pass O(instructions) and, unlike a worklist that
// inserts temporary `trunc`/`zext` pairs at every not-yet-legalized boundary,
// it never materialises intermediate illegal-width values that would have to
// be cleaned up afterwards.
//
// All-or-nothing legality check
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Phase 2 only inspects the IR; it does not modify it. Every instruction that
// defines or consumes an illegal integer type is checked against the set of
// supported opcodes and operand kinds, and the whole function is rejected if
// any of them fails. Doing this up front guarantees that phase 3 either
// rewrites the function completely or does not start at all, so the pass never
// leaves the function in a half-legalized state that is harder to diagnose
// than the original IR. Phase 0 runs before this check because it is
// self-contained and locally valid regardless of the rest of the function.
//
// Teardown order
// ~~~~~~~~~~~~~~
// Original instructions cannot simply be erased as they are rewritten: they
// may still be referenced by other originals that have not been processed yet,
// and phi nodes in loops form reference cycles. Phase 4 therefore first calls
// `dropAllReferences()` on all collected instructions (breaking every cycle
// and dropping all uses among them) and only then erases them. Both loops run
// in reverse work-list order so that definitions are destroyed after their
// users. Instructions whose result type was already legal - e.g. an `icmp i48`
// producing `i1` - are RAUW'd onto their replacement during phase 3, since
// those may have users outside the work list.
//
// Known limitations
// ~~~~~~~~~~~~~~~~~
// RPO only visits blocks reachable from the entry block, so illegal types in
// unreachable code are not rewritten. Such code is normally removed by the
// preceding optimization pipeline; if it survives, phase 6 reports it, as that
// sweep iterates over all basic blocks.

#include "hipSYCL/compiler/llvm-to-backend/clspv/LegalizeIntWidthsPass.hpp"
#include "hipSYCL/common/debug.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueHandle.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Transforms/Utils/Local.h>

namespace hipsycl {
namespace compiler {

namespace {

bool isLegalWidth(unsigned BitWidth) {
  return BitWidth == 1 || BitWidth == 8 || BitWidth == 16 || BitWidth == 32 ||
         BitWidth == 64;
}

bool isIllegalIntTy(const llvm::Type *T) {
  return T->isIntegerTy() && !isLegalWidth(T->getIntegerBitWidth());
}

/// Smallest legal width that can hold \p BitWidth bits, 0 if there is none.
unsigned getPromotedWidth(unsigned BitWidth) {
  for (unsigned W : {8u, 16u, 32u, 64u}) {
    if (BitWidth <= W)
      return W;
  }
  return 0;
}

llvm::IntegerType *getPromotedTy(llvm::Type *T) {
  unsigned W = getPromotedWidth(T->getIntegerBitWidth());
  assert(W != 0 && "No legal type to promote to");
  return llvm::IntegerType::get(T->getContext(), W);
}

struct MemoryChunk {
  uint64_t OffsetInBytes;
  unsigned Bits;
};

/// Decomposes an object of \p BitWidth bits into chunks of legal width,
/// starting with the widest chunk. The offset of every chunk is a multiple of
/// its own size, so that each access can be expressed as a typed GEP.
llvm::SmallVector<MemoryChunk, 4> computeChunks(unsigned BitWidth) {
  llvm::SmallVector<MemoryChunk, 4> Chunks;
  uint64_t RemainingBytes = (BitWidth + 7) / 8;
  uint64_t Offset = 0;
  while (RemainingBytes > 0) {
    uint64_t ChunkBytes = 8;
    while (ChunkBytes > RemainingBytes)
      ChunkBytes /= 2;
    Chunks.push_back({Offset, static_cast<unsigned>(ChunkBytes * 8)});
    Offset += ChunkBytes;
    RemainingBytes -= ChunkBytes;
  }
  return Chunks;
}

llvm::Value *getChunkPtr(llvm::IRBuilder<> &B, llvm::Value *BasePtr,
                         const MemoryChunk &C, llvm::Type *ChunkTy) {
  if (C.OffsetInBytes == 0)
    return BasePtr;
  const uint64_t ChunkBytes = C.Bits / 8;
  assert(C.OffsetInBytes % ChunkBytes == 0 && "Chunk offset not aligned");
  return B.CreateConstInBoundsGEP1_64(ChunkTy, BasePtr,
                                      C.OffsetInBytes / ChunkBytes);
}

llvm::Align getChunkAlign(llvm::Align BaseAlign, const MemoryChunk &C) {
  if (C.OffsetInBytes == 0)
    return BaseAlign;
  return llvm::commonAlignment(BaseAlign, C.OffsetInBytes);
}

bool isSupportedOpcode(const llvm::Instruction *I) {
  switch (I->getOpcode()) {
  case llvm::Instruction::Load:
    return llvm::cast<llvm::LoadInst>(I)->isSimple();
  case llvm::Instruction::Store:
    return llvm::cast<llvm::StoreInst>(I)->isSimple();
  case llvm::Instruction::PHI:
  case llvm::Instruction::Select:
  case llvm::Instruction::ICmp:
  case llvm::Instruction::Trunc:
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt:
  case llvm::Instruction::Add:
  case llvm::Instruction::Sub:
  case llvm::Instruction::Mul:
  case llvm::Instruction::And:
  case llvm::Instruction::Or:
  case llvm::Instruction::Xor:
  case llvm::Instruction::Shl:
  case llvm::Instruction::LShr:
  case llvm::Instruction::AShr:
  case llvm::Instruction::UDiv:
  case llvm::Instruction::SDiv:
  case llvm::Instruction::URem:
  case llvm::Instruction::SRem:
    return true;
  default:
    return false;
  }
}

/// Every illegal type the pass can rewrite must be a scalar integer that fits
/// into a legal type. Illegal integers nested in vectors or wider than 64 bits
/// are not supported by the promotion logic.
bool isPromotableIllegalTy(const llvm::Type *T) {
  return isIllegalIntTy(T) && getPromotedWidth(T->getIntegerBitWidth()) != 0;
}

bool hasIllegalOperand(const llvm::Instruction *I) {
  for (const llvm::Use &U : I->operands()) {
    if (isIllegalIntTy(U.get()->getType()))
      return true;
  }
  return false;
}

class IntWidthLegalizer {
public:
  IntWidthLegalizer(llvm::Function &F)
      : F{F}, DL{F.getParent()->getDataLayout()} {}

  /// Returns true if the function was modified.
  bool run();

private:
  /// Replaces stores of values too wide to be promoted by a sequence of chunk
  /// sized stores.
  bool splitWideStores();

  /// Materializes the bits [\p LoBit, \p LoBit + \p Bits) of \p V as a value
  /// of type i\p Bits, without ever creating a value of illegal width.
  ///
  /// Returns null if the bits cannot be extracted, in which case no
  /// instructions have been added that are not trivially dead.
  llvm::Value *extractBits(llvm::IRBuilder<> &B, llvm::Value *V,
                           unsigned LoBit, unsigned Bits);

  bool collectWork();
  bool rewrite();

  /// Value of promoted type whose low `N` bits match the low `N` bits of \p V.
  /// Bits above `N` are undefined.
  llvm::Value *getPromoted(llvm::Value *V);

  llvm::Value *maskToWidth(llvm::IRBuilder<> &B, llvm::Value *PromotedV,
                           unsigned Width);
  llvm::Value *signExtendWithin(llvm::IRBuilder<> &B, llvm::Value *PromotedV,
                                unsigned Width);
  /// Adjusts a promoted value to \p TargetTy, preserving its low bits.
  llvm::Value *adjustWidth(llvm::IRBuilder<> &B, llvm::Value *V,
                           llvm::Type *TargetTy);

  llvm::Value *emitSplitLoad(llvm::IRBuilder<> &B, llvm::LoadInst *LI,
                             llvm::Type *CombineTy);
  void emitSplitStore(llvm::IRBuilder<> &B, llvm::StoreInst *SI,
                      llvm::Value *PromotedVal);

  llvm::Value *rewriteInstruction(llvm::Instruction *I);

  /// Makes sure that phi nodes created by the pass are well formed, even if
  /// legalization was aborted half way through.
  void completeCreatedPhis();

  llvm::Function &F;
  const llvm::DataLayout &DL;

  llvm::SmallVector<llvm::Instruction *, 32> Work;
  llvm::SmallVector<llvm::AllocaInst *, 8> IllegalAllocas;
  llvm::DenseMap<llvm::Value *, llvm::Value *> Promoted;
  llvm::DenseMap<llvm::PHINode *, llvm::PHINode *> PhiMap;
};

llvm::Value *IntWidthLegalizer::extractBits(llvm::IRBuilder<> &B,
                                           llvm::Value *V, unsigned LoBit,
                                           unsigned Bits) {
  auto *ResTy = llvm::IntegerType::get(F.getContext(), Bits);
  const unsigned Width = V->getType()->getIntegerBitWidth();
  assert(LoBit + Bits <= Width && "Requested bits outside of value");

  if (llvm::isa<llvm::UndefValue>(V))
    return llvm::UndefValue::get(ResTy);

  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(V))
    return llvm::ConstantInt::get(ResTy, CI->getValue().lshr(LoBit).trunc(Bits));

  // A plain load of a wide value: load the requested chunk directly.
  if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(V)) {
    if (!LI->isSimple() || LoBit % 8 != 0 || Bits % 8 != 0 ||
        !isLegalWidth(Bits))
      return nullptr;
    const MemoryChunk C{LoBit / 8, Bits};
    if (C.OffsetInBytes % (Bits / 8) != 0)
      return nullptr;
    llvm::IRBuilder<> LB{LI};
    llvm::Value *Ptr = getChunkPtr(LB, LI->getPointerOperand(), C, ResTy);
    return LB.CreateAlignedLoad(ResTy, Ptr, getChunkAlign(LI->getAlign(), C));
  }

  if (auto *I = llvm::dyn_cast<llvm::Instruction>(V)) {
    switch (I->getOpcode()) {
    case llvm::Instruction::ZExt: {
      // Bits at or above the source width are known to be zero.
      llvm::Value *Src = I->getOperand(0);
      const unsigned SrcWidth = Src->getType()->getIntegerBitWidth();
      if (LoBit >= SrcWidth)
        return llvm::ConstantInt::get(ResTy, 0);
      const unsigned SubBits = std::min(Bits, SrcWidth - LoBit);
      if (!isLegalWidth(SubBits))
        return nullptr;
      llvm::Value *Sub = extractBits(B, Src, LoBit, SubBits);
      if (!Sub)
        return nullptr;
      return SubBits == Bits ? Sub : B.CreateZExt(Sub, ResTy);
    }
    case llvm::Instruction::Trunc: {
      llvm::Value *Src = I->getOperand(0);
      // Truncation only drops high bits, so the requested bits are unchanged.
      return extractBits(B, Src, LoBit, Bits);
    }
    case llvm::Instruction::Shl: {
      auto *Amount = llvm::dyn_cast<llvm::ConstantInt>(I->getOperand(1));
      if (!Amount)
        return nullptr;
      const uint64_t Sh = Amount->getZExtValue();
      // Bits below the shift amount are zero.
      if (LoBit + Bits <= Sh)
        return llvm::ConstantInt::get(ResTy, 0);
      if (LoBit >= Sh)
        return extractBits(B, I->getOperand(0), LoBit - Sh, Bits);
      llvm::Value *Sub = extractBits(B, I->getOperand(0), 0, Bits);
      if (!Sub)
        return nullptr;
      return B.CreateShl(Sub, llvm::ConstantInt::get(ResTy, Sh - LoBit));
    }
    case llvm::Instruction::LShr: {
      auto *Amount = llvm::dyn_cast<llvm::ConstantInt>(I->getOperand(1));
      if (!Amount)
        return nullptr;
      const uint64_t Sh = Amount->getZExtValue();
      if (LoBit + Sh >= Width)
        return llvm::ConstantInt::get(ResTy, 0);
      if (LoBit + Bits + Sh > Width)
        return nullptr;
      return extractBits(B, I->getOperand(0), LoBit + Sh, Bits);
    }
    case llvm::Instruction::And:
    case llvm::Instruction::Or:
    case llvm::Instruction::Xor: {
      // Bitwise operations act on each bit independently.
      llvm::Value *L = extractBits(B, I->getOperand(0), LoBit, Bits);
      if (!L)
        return nullptr;
      llvm::Value *R = extractBits(B, I->getOperand(1), LoBit, Bits);
      if (!R)
        return nullptr;
      return B.CreateBinOp(
          static_cast<llvm::Instruction::BinaryOps>(I->getOpcode()), L, R);
    }
    case llvm::Instruction::Select: {
      llvm::Value *T = extractBits(B, I->getOperand(1), LoBit, Bits);
      if (!T)
        return nullptr;
      llvm::Value *Fa = extractBits(B, I->getOperand(2), LoBit, Bits);
      if (!Fa)
        return nullptr;
      return B.CreateSelect(I->getOperand(0), T, Fa);
    }
    default:
      break;
    }
  }

  // Opaque value: it can only be taken apart if it is of legal width itself.
  if (!isLegalWidth(Width))
    return nullptr;
  llvm::Value *Shifted =
      LoBit == 0 ? V
                 : B.CreateLShr(V, llvm::ConstantInt::get(V->getType(), LoBit));
  return Width == Bits ? Shifted : B.CreateTrunc(Shifted, ResTy);
}

// Phase 0: Rewrite stores of values that are too wide to be promoted.
//
// Values wider than 64 bits have no legal type they could be promoted to, so
// the only way to legalize them is to never materialize them: the stored value
// is decomposed into chunks of legal width that are stored individually. This
// works whenever every chunk can be computed without constructing a wide value
// itself, which `extractBits()` checks.
//
// Typical sources of such values are `sycl::vec<float, 3/4>` return values of
// relational builtins, which SROA packs into a single `i128` built from a tree
// of `zext`/`shl`/`or`, and plain copies of wide objects.
bool IntWidthLegalizer::splitWideStores() {
  llvm::SmallVector<llvm::StoreInst *, 8> WideStores;
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I);
      if (!SI || !SI->isSimple())
        continue;
      llvm::Type *ValTy = SI->getValueOperand()->getType();
      if (isIllegalIntTy(ValTy) &&
          getPromotedWidth(ValTy->getIntegerBitWidth()) == 0)
        WideStores.push_back(SI);
    }
  }

  bool Changed = false;
  llvm::SmallVector<llvm::WeakTrackingVH, 16> MaybeDead;
  for (llvm::StoreInst *SI : WideStores) {
    llvm::Value *Val = SI->getValueOperand();
    const unsigned BitWidth = Val->getType()->getIntegerBitWidth();
    auto Chunks = computeChunks(BitWidth);

    // Build the chunk values first: if any of them cannot be extracted the
    // store is left alone, and the instructions created so far are dead and
    // cleaned up below.
    llvm::IRBuilder<> B{SI};
    llvm::SmallVector<llvm::Value *, 4> ChunkValues;
    bool Extracted = true;
    for (const MemoryChunk &C : Chunks) {
      llvm::Value *ChunkVal = extractBits(B, Val, C.OffsetInBytes * 8, C.Bits);
      if (!ChunkVal) {
        Extracted = false;
        break;
      }
      ChunkValues.push_back(ChunkVal);
    }

    for (llvm::Value *V : ChunkValues)
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(V))
        MaybeDead.push_back(I);

    if (!Extracted)
      continue;

    for (size_t i = 0; i < Chunks.size(); ++i) {
      const MemoryChunk &C = Chunks[i];
      auto *ChunkTy = llvm::IntegerType::get(F.getContext(), C.Bits);
      llvm::Value *DstPtr = getChunkPtr(B, SI->getPointerOperand(), C, ChunkTy);
      B.CreateAlignedStore(ChunkValues[i], DstPtr, getChunkAlign(SI->getAlign(), C));
    }

    if (auto *ValI = llvm::dyn_cast<llvm::Instruction>(Val))
      MaybeDead.push_back(ValI);
    SI->eraseFromParent();
    Changed = true;
  }

  // Remove the now dead wide value computations, as they would otherwise still
  // contain illegal types and cause the legalization below to bail out.
  llvm::RecursivelyDeleteTriviallyDeadInstructionsPermissive(MaybeDead);

  return Changed;
}

// Phase 2: Scan the function in reverse post-order and record every
// instruction that needs rewriting, without modifying the IR.
//
// Returns false if the function contains an illegal integer type the pass
// cannot handle. In that case the caller abandons legalization of this
// function entirely, see the "All-or-nothing legality check" note at the top
// of this file. Rejected are:
//
//  * opcodes outside the supported set, including volatile/atomic accesses
//    and calls taking or returning an illegal type,
//  * illegal types that do not fit into a legal one (>64 bits); these are
//    only handled by phase 0 for plain copies,
//  * illegal-typed operands that are neither instructions, constant ints nor
//    undef, as there would be no way to derive a promoted counterpart.
//
// Allocas are collected separately: they are rewritten in phase 5 whether or
// not the alloca itself appears in the work list, and their illegal type never
// propagates into a value, since it is only part of the allocated type.
bool IntWidthLegalizer::collectWork() {
  llvm::ReversePostOrderTraversal<llvm::Function *> RPO{&F};
  for (llvm::BasicBlock *BB : RPO) {
    for (llvm::Instruction &I : *BB) {
      if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        if (isIllegalIntTy(AI->getAllocatedType()))
          IllegalAllocas.push_back(AI);
      }

      if (!isIllegalIntTy(I.getType()) && !hasIllegalOperand(&I))
        continue;

      if (!isSupportedOpcode(&I)) {
        HIPSYCL_DEBUG_WARNING
            << "LegalizeIntWidthsPass: Unsupported instruction using an "
               "integer type of non-standard bit width in function "
            << F.getName().str() << "; clspv may reject this kernel\n";
        return false;
      }

      auto reject = [&](const char *Reason) {
        HIPSYCL_DEBUG_WARNING
            << "LegalizeIntWidthsPass: " << Reason << " in function "
            << F.getName().str() << " (" << I.getOpcodeName()
            << "); clspv may reject this kernel\n";
      };

      // The promotion logic only supports scalar integers that fit into a
      // legal type.
      if (isIllegalIntTy(I.getType()) && !isPromotableIllegalTy(I.getType())) {
        reject("Integer type too wide to be promoted to a legal width");
        return false;
      }
      for (llvm::Use &U : I.operands()) {
        llvm::Type *OpTy = U.get()->getType();
        if (isIllegalIntTy(OpTy) && !isPromotableIllegalTy(OpTy)) {
          reject("Operand of an integer type too wide to be promoted to a "
                 "legal width");
          return false;
        }
        if (isIllegalIntTy(OpTy) && !llvm::isa<llvm::Instruction>(U.get()) &&
            !llvm::isa<llvm::ConstantInt>(U.get()) &&
            !llvm::isa<llvm::UndefValue>(U.get())) {
          reject("Operand of non-standard integer width with no legalizable "
                 "counterpart");
          return false;
        }
      }

      Work.push_back(&I);
    }
  }
  return true;
}

// Returns the legal-width counterpart of an illegal-width value.
//
// For instructions this is the entry recorded by phase 3; thanks to the RPO
// order it is already present for every operand, with the exception of phi
// back edges which are resolved after the main loop. Constants and undef are
// materialised on demand. A null return means the value cannot be legalized
// and aborts the rewrite.
llvm::Value *IntWidthLegalizer::getPromoted(llvm::Value *V) {
  assert(isIllegalIntTy(V->getType()) && "Value is already of legal type");

  auto It = Promoted.find(V);
  if (It != Promoted.end())
    return It->second;

  llvm::IntegerType *PromTy = getPromotedTy(V->getType());
  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(V))
    return llvm::ConstantInt::get(PromTy,
                                  CI->getValue().zext(PromTy->getBitWidth()));
  if (llvm::isa<llvm::UndefValue>(V))
    return llvm::UndefValue::get(PromTy);

  return nullptr;
}

llvm::Value *IntWidthLegalizer::maskToWidth(llvm::IRBuilder<> &B,
                                            llvm::Value *PromotedV,
                                            unsigned Width) {
  const unsigned PromWidth = PromotedV->getType()->getIntegerBitWidth();
  if (PromWidth == Width)
    return PromotedV;
  return B.CreateAnd(PromotedV,
                     llvm::ConstantInt::get(
                         PromotedV->getType(),
                         llvm::APInt::getLowBitsSet(PromWidth, Width)));
}

llvm::Value *IntWidthLegalizer::signExtendWithin(llvm::IRBuilder<> &B,
                                                 llvm::Value *PromotedV,
                                                 unsigned Width) {
  const unsigned PromWidth = PromotedV->getType()->getIntegerBitWidth();
  if (PromWidth == Width)
    return PromotedV;
  llvm::Value *ShiftAmount =
      llvm::ConstantInt::get(PromotedV->getType(), PromWidth - Width);
  return B.CreateAShr(B.CreateShl(PromotedV, ShiftAmount), ShiftAmount);
}

llvm::Value *IntWidthLegalizer::adjustWidth(llvm::IRBuilder<> &B,
                                            llvm::Value *V,
                                            llvm::Type *TargetTy) {
  const unsigned FromWidth = V->getType()->getIntegerBitWidth();
  const unsigned ToWidth = TargetTy->getIntegerBitWidth();
  if (FromWidth == ToWidth)
    return V;
  if (FromWidth > ToWidth)
    return B.CreateTrunc(V, TargetTy);
  return B.CreateZExt(V, TargetTy);
}

llvm::Value *IntWidthLegalizer::emitSplitLoad(llvm::IRBuilder<> &B,
                                              llvm::LoadInst *LI,
                                              llvm::Type *CombineTy) {
  const unsigned BitWidth = LI->getType()->getIntegerBitWidth();
  auto Chunks = computeChunks(BitWidth);

  llvm::Value *Result = nullptr;
  for (const MemoryChunk &C : Chunks) {
    auto *ChunkTy = llvm::IntegerType::get(F.getContext(), C.Bits);
    llvm::Value *Ptr = getChunkPtr(B, LI->getPointerOperand(), C, ChunkTy);
    llvm::Value *ChunkVal =
        B.CreateAlignedLoad(ChunkTy, Ptr, getChunkAlign(LI->getAlign(), C));
    llvm::Value *Extended = adjustWidth(B, ChunkVal, CombineTy);
    if (C.OffsetInBytes != 0)
      Extended = B.CreateShl(
          Extended, llvm::ConstantInt::get(CombineTy, C.OffsetInBytes * 8));
    Result = Result ? B.CreateOr(Result, Extended) : Extended;
  }
  return Result;
}

void IntWidthLegalizer::emitSplitStore(llvm::IRBuilder<> &B,
                                       llvm::StoreInst *SI,
                                       llvm::Value *PromotedVal) {
  const unsigned BitWidth =
      SI->getValueOperand()->getType()->getIntegerBitWidth();
  auto Chunks = computeChunks(BitWidth);

  // Bits above the original width are undefined in the promoted value; mask
  // them off so that no garbage is written to memory.
  llvm::Value *Val = maskToWidth(B, PromotedVal, BitWidth);
  for (const MemoryChunk &C : Chunks) {
    auto *ChunkTy = llvm::IntegerType::get(F.getContext(), C.Bits);
    llvm::Value *Ptr = getChunkPtr(B, SI->getPointerOperand(), C, ChunkTy);
    llvm::Value *ChunkVal = Val;
    if (C.OffsetInBytes != 0)
      ChunkVal = B.CreateLShr(
          ChunkVal, llvm::ConstantInt::get(Val->getType(), C.OffsetInBytes * 8));
    ChunkVal = adjustWidth(B, ChunkVal, ChunkTy);
    B.CreateAlignedStore(ChunkVal, Ptr, getChunkAlign(SI->getAlign(), C));
  }
}

llvm::Value *IntWidthLegalizer::rewriteInstruction(llvm::Instruction *I) {
  llvm::IRBuilder<> B{I};

  auto promotedOperand = [&](unsigned Idx) -> llvm::Value * {
    llvm::Value *Op = I->getOperand(Idx);
    return isIllegalIntTy(Op->getType()) ? getPromoted(Op) : Op;
  };

  switch (I->getOpcode()) {
  case llvm::Instruction::Load: {
    auto *LI = llvm::cast<llvm::LoadInst>(I);
    return emitSplitLoad(B, LI, getPromotedTy(LI->getType()));
  }
  case llvm::Instruction::Store: {
    auto *SI = llvm::cast<llvm::StoreInst>(I);
    llvm::Value *Val = promotedOperand(0);
    if (!Val)
      return nullptr;
    emitSplitStore(B, SI, Val);
    return nullptr;
  }
  case llvm::Instruction::PHI: {
    // Incoming values are patched once all instructions have been rewritten.
    auto *Phi = llvm::cast<llvm::PHINode>(I);
    auto *NewPhi = llvm::PHINode::Create(getPromotedTy(Phi->getType()),
                                         Phi->getNumIncomingValues(),
                                         Phi->getName() + ".legalized", Phi);
    PhiMap[Phi] = NewPhi;
    return NewPhi;
  }
  case llvm::Instruction::Select: {
    llvm::Value *T = promotedOperand(1);
    llvm::Value *Fa = promotedOperand(2);
    if (!T || !Fa)
      return nullptr;
    return B.CreateSelect(I->getOperand(0), T, Fa);
  }
  case llvm::Instruction::ICmp: {
    auto *Cmp = llvm::cast<llvm::ICmpInst>(I);
    const unsigned Width = Cmp->getOperand(0)->getType()->getIntegerBitWidth();
    llvm::Value *L = promotedOperand(0);
    llvm::Value *R = promotedOperand(1);
    if (!L || !R)
      return nullptr;
    if (Cmp->isSigned()) {
      L = signExtendWithin(B, L, Width);
      R = signExtendWithin(B, R, Width);
    } else {
      L = maskToWidth(B, L, Width);
      R = maskToWidth(B, R, Width);
    }
    return B.CreateICmp(Cmp->getPredicate(), L, R);
  }
  case llvm::Instruction::Trunc:
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt: {
    llvm::Value *Src = promotedOperand(0);
    if (!Src)
      return nullptr;
    llvm::Type *SrcTy = I->getOperand(0)->getType();
    llvm::Type *DstTy = I->getType();
    llvm::Type *NewDstTy =
        isIllegalIntTy(DstTy) ? getPromotedTy(DstTy) : DstTy;

    if (isIllegalIntTy(SrcTy)) {
      const unsigned SrcWidth = SrcTy->getIntegerBitWidth();
      if (I->getOpcode() == llvm::Instruction::ZExt)
        Src = maskToWidth(B, Src, SrcWidth);
      else if (I->getOpcode() == llvm::Instruction::SExt)
        Src = signExtendWithin(B, Src, SrcWidth);
    }

    const unsigned SrcPromWidth = Src->getType()->getIntegerBitWidth();
    const unsigned DstPromWidth = NewDstTy->getIntegerBitWidth();
    if (SrcPromWidth == DstPromWidth)
      return Src;
    if (SrcPromWidth > DstPromWidth)
      return B.CreateTrunc(Src, NewDstTy);
    // Widening a legal source keeps the original signedness, widening an
    // already sign/zero extended illegal source can use either.
    if (I->getOpcode() == llvm::Instruction::SExt)
      return B.CreateSExt(Src, NewDstTy);
    return B.CreateZExt(Src, NewDstTy);
  }
  default:
    break;
  }

  auto *BinOp = llvm::dyn_cast<llvm::BinaryOperator>(I);
  if (!BinOp)
    return nullptr;

  const unsigned Width = BinOp->getType()->getIntegerBitWidth();
  llvm::Value *L = promotedOperand(0);
  llvm::Value *R = promotedOperand(1);
  if (!L || !R)
    return nullptr;

  switch (BinOp->getOpcode()) {
  case llvm::Instruction::Add:
  case llvm::Instruction::Sub:
  case llvm::Instruction::Mul:
  case llvm::Instruction::And:
  case llvm::Instruction::Or:
  case llvm::Instruction::Xor:
  case llvm::Instruction::Shl:
    // The low 'Width' bits of the result only depend on the low 'Width' bits
    // of the operands.
    break;
  case llvm::Instruction::LShr:
  case llvm::Instruction::UDiv:
  case llvm::Instruction::URem:
    L = maskToWidth(B, L, Width);
    R = maskToWidth(B, R, Width);
    break;
  case llvm::Instruction::AShr:
    L = signExtendWithin(B, L, Width);
    R = maskToWidth(B, R, Width);
    break;
  case llvm::Instruction::SDiv:
  case llvm::Instruction::SRem:
    L = signExtendWithin(B, L, Width);
    R = signExtendWithin(B, R, Width);
    break;
  default:
    return nullptr;
  }

  return B.CreateBinOp(BinOp->getOpcode(), L, R);
}

void IntWidthLegalizer::completeCreatedPhis() {
  for (auto &Entry : PhiMap) {
    llvm::PHINode *Old = Entry.first;
    llvm::PHINode *New = Entry.second;
    while (New->getNumIncomingValues() < Old->getNumIncomingValues()) {
      const unsigned Idx = New->getNumIncomingValues();
      New->addIncoming(llvm::PoisonValue::get(New->getType()),
                       Old->getIncomingBlock(Idx));
    }
  }
}

// Phase 3 and 4: rewrite all collected instructions in RPO order, patch the
// phi nodes created along the way, then tear down the originals.
bool IntWidthLegalizer::rewrite() {
  // Phase 3. Operands are guaranteed to be legalized already (RPO), so a
  // single ordered pass is sufficient; no fixpoint iteration is required.
  for (llvm::Instruction *I : Work) {
    llvm::Value *New = rewriteInstruction(I);
    if (!New) {
      if (!I->getType()->isVoidTy()) {
        HIPSYCL_DEBUG_WARNING
            << "LegalizeIntWidthsPass: Could not legalize instruction in "
               "function "
            << F.getName().str() << "; clspv may reject this kernel\n";
        completeCreatedPhis();
        return false;
      }
      continue;
    }
    if (isIllegalIntTy(I->getType()))
      Promoted[I] = New;
    else
      I->replaceAllUsesWith(New);
  }

  // Loop back edges are the only place where RPO does not guarantee that an
  // operand has been rewritten before its user, so the new phi nodes were
  // created with no incoming values. Now that every value has a legalized
  // counterpart, fill them in.
  for (auto &Entry : PhiMap) {
    llvm::PHINode *Old = Entry.first;
    llvm::PHINode *New = Entry.second;
    for (unsigned i = 0; i < Old->getNumIncomingValues(); ++i) {
      llvm::Value *In = Old->getIncomingValue(i);
      llvm::Value *NewIn = isIllegalIntTy(In->getType()) ? getPromoted(In) : In;
      if (!NewIn) {
        completeCreatedPhis();
        return false;
      }
      New->addIncoming(NewIn, Old->getIncomingBlock(i));
    }
  }

  // Phase 4. The originals may still reference each other, and phi nodes in
  // loops form reference cycles, so drop all references first to break those
  // cycles before erasing. Both loops run in reverse work-list order, i.e.
  // users before definitions.
  for (llvm::Instruction *I : llvm::reverse(Work))
    I->dropAllReferences();
  for (llvm::Instruction *I : llvm::reverse(Work))
    I->eraseFromParent();

  return true;
}

bool IntWidthLegalizer::run() {
  // Phase 0. Values wider than 64 bits have no legal type to be promoted to
  // and are therefore handled by decomposing the stored value into chunks of
  // legal width. This is independent of the legality check below.
  bool Changed = splitWideStores();

  // Phase 1. Illegal types in the signature would require rewriting the
  // function itself along with all of its call sites, which is out of scope:
  // by the time this pass runs the backend has aggressively inlined the
  // module, so such signatures are not expected in practice.
  auto rejectSignature = [&]() {
    HIPSYCL_DEBUG_WARNING
        << "LegalizeIntWidthsPass: Signature of function " << F.getName().str()
        << " uses an integer type of non-standard bit width; clspv may reject "
           "this kernel\n";
  };
  for (const llvm::Argument &A : F.args()) {
    if (isIllegalIntTy(A.getType())) {
      rejectSignature();
      return Changed;
    }
  }
  if (isIllegalIntTy(F.getReturnType())) {
    rejectSignature();
    return Changed;
  }

  // Phase 2.
  if (!collectWork())
    return Changed;

  // Phase 3 and 4.

  if (!Work.empty()) {
    if (!rewrite()) {
      HIPSYCL_DEBUG_WARNING
          << "LegalizeIntWidthsPass: Failed to legalize integer widths in "
             "function "
          << F.getName().str() << "\n";
    }
    Changed = true;
  }

  // Phase 5. Allocas of illegal types are replaced by equally sized byte
  // arrays; all accesses to them have been split into legal accesses above.
  // Since pointers are opaque, users of the alloca do not need to be rewritten
  // beyond that.
  for (llvm::AllocaInst *AI : IllegalAllocas) {
    const uint64_t SizeInBytes = DL.getTypeStoreSize(AI->getAllocatedType());
    llvm::IRBuilder<> B{AI};
    auto *NewTy = llvm::ArrayType::get(llvm::Type::getInt8Ty(F.getContext()),
                                       SizeInBytes);
    auto *NewAlloca = B.CreateAlloca(NewTy, AI->getAddressSpace(),
                                     AI->getArraySize(), AI->getName());
    NewAlloca->setAlignment(AI->getAlign());
    AI->replaceAllUsesWith(NewAlloca);
    AI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

/// Phase 6: Reports integer types of non-standard width that survived
/// legalization, as those would result in an obscure clspv error message later
/// on. Unlike the RPO walk of phase 2 this visits all basic blocks, so it also
/// covers unreachable code the rewrite does not reach.
void warnOnRemainingIllegalTypes(llvm::Function &F) {
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      bool Illegal = isIllegalIntTy(I.getType());
      for (llvm::Use &U : I.operands())
        Illegal |= isIllegalIntTy(U.get()->getType());
      if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(&I))
        Illegal |= isIllegalIntTy(AI->getAllocatedType());

      if (Illegal) {
        HIPSYCL_DEBUG_WARNING
            << "LegalizeIntWidthsPass: Function " << F.getName().str()
            << " still contains integer types of non-standard bit width, "
               "which clspv cannot lower to SPIR-V\n";
        return;
      }
    }
  }
}

} // namespace

llvm::PreservedAnalyses
LegalizeIntWidthsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
  if (F.isDeclaration())
    return llvm::PreservedAnalyses::all();

  IntWidthLegalizer Legalizer{F};
  const bool Changed = Legalizer.run();

  warnOnRemainingIllegalTypes(F);

  return Changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
