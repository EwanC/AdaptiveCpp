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
#include "hipSYCL/compiler/llvm-to-backend/clspv/TypedGEPRecoveryPass.hpp"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/KnownBits.h>
#include <llvm/Support/MathExtras.h>

namespace hipsycl {
namespace compiler {

namespace {

bool isDynamicByteGEP(llvm::GetElementPtrInst *GEP) {
  return GEP->getSourceElementType()->isIntegerTy(8) &&
         GEP->getNumIndices() == 1 &&
         !llvm::isa<llvm::Constant>(GEP->getOperand(1));
}

// Collects the types of all loads/stores reachable from Root through
// byte GEPs, PHIs and selects. Returns false if the pointer is used in any
// other way (escapes, is passed to calls, ...).
bool collectAccessTypes(llvm::Value *Root,
                        llvm::SmallPtrSetImpl<llvm::Type *> &Types) {
  llvm::SmallVector<llvm::Value *, 16> Worklist{Root};
  llvm::SmallPtrSet<llvm::Value *, 16> Visited{Root};

  while (!Worklist.empty()) {
    llvm::Value *V = Worklist.pop_back_val();
    for (llvm::User *U : V->users()) {
      if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U)) {
        Types.insert(LI->getType());
      } else if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(U)) {
        if (SI->getPointerOperand() != V)
          return false;
        Types.insert(SI->getValueOperand()->getType());
      } else if (llvm::isa<llvm::ICmpInst>(U)) {
        continue;
      } else if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(U)) {
        if (GEP->getPointerOperand() != V ||
            !GEP->getSourceElementType()->isIntegerTy(8) ||
            GEP->getNumIndices() != 1)
          return false;
        if (Visited.insert(GEP).second)
          Worklist.push_back(GEP);
      } else if (llvm::isa<llvm::PHINode, llvm::SelectInst>(U)) {
        if (Visited.insert(U).second)
          Worklist.push_back(U);
      } else {
        return false;
      }
    }
  }
  return !Types.empty();
}

struct Rewrite {
  llvm::GetElementPtrInst *GEP;
  llvm::Type *ElemTy;
  unsigned Shift;
};

} // namespace

llvm::PreservedAnalyses
TypedGEPRecoveryPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
  const llvm::DataLayout &DL = F.getParent()->getDataLayout();

  // Decide on all rewrites before changing anything, so that the analysis of
  // one GEP is not affected by the rewrite of another GEP in the same web.
  llvm::SmallVector<Rewrite, 16> Rewrites;
  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I);
      if (!GEP || !isDynamicByteGEP(GEP))
        continue;

      llvm::SmallPtrSet<llvm::Type *, 4> Types;
      if (!collectAccessTypes(GEP, Types))
        continue;

      uint64_t Size = 0;
      bool Consistent = true;
      for (llvm::Type *T : Types) {
        if (!T->isSized()) {
          Consistent = false;
          break;
        }
        uint64_t S = DL.getTypeStoreSize(T).getFixedValue();
        if (Size != 0 && S != Size)
          Consistent = false;
        Size = S;
      }
      if (!Consistent || Size < 2 || Size > 8 || !llvm::isPowerOf2_64(Size))
        continue;

      unsigned Shift = llvm::Log2_64(Size);
      llvm::KnownBits Known = llvm::computeKnownBits(GEP->getOperand(1), DL);
      if (Known.countMinTrailingZeros() < Shift)
        continue;

      llvm::Type *ElemTy =
          Types.size() == 1 ? *Types.begin()
                            : llvm::IntegerType::get(F.getContext(), Size * 8);
      Rewrites.push_back({GEP, ElemTy, Shift});
    }
  }

  if (Rewrites.empty())
    return llvm::PreservedAnalyses::all();

  for (auto &R : Rewrites) {
    llvm::IRBuilder<> B(R.GEP);
    llvm::Value *Idx = B.CreateAShr(R.GEP->getOperand(1), R.Shift, "gep.idx",
                                    /*isExact=*/true);
    llvm::Value *NewGEPVal = B.CreateGEP(R.ElemTy, R.GEP->getPointerOperand(),
                                         {Idx}, R.GEP->getName());
    // The scaled index reproduces the original offset exactly, so inbounds
    // still holds. nuw is dropped, as it would not hold for negative offsets.
    if (auto *NewGEP = llvm::dyn_cast<llvm::GetElementPtrInst>(NewGEPVal))
      NewGEP->setIsInBounds(R.GEP->isInBounds());
    R.GEP->replaceAllUsesWith(NewGEPVal);
    R.GEP->eraseFromParent();
  }

  return llvm::PreservedAnalyses::none();
}

} // namespace compiler
} // namespace hipsycl
