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
#include <llvm/Support/Alignment.h>

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

// Smallest legal width that can hold 'BitWidth' bits, 0 if there is none.
unsigned getPromotedWidth(unsigned BitWidth) {
  for (unsigned W : {8u, 16u, 32u, 64u}) {
    if (BitWidth <= W) {
      return W;
    }
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

// Decomposes an object of 'BitWidth' bits into chunks of legal width,
// starting with the widest chunk. The offset of every chunk is a multiple of
// its own size, so that each access can be expressed as a typed GEP.
llvm::SmallVector<MemoryChunk, 4> computeChunks(unsigned BitWidth) {
  llvm::SmallVector<MemoryChunk, 4> Chunks;
  uint64_t RemainingBytes = (BitWidth + 7) / 8;
  uint64_t Offset = 0;
  while (RemainingBytes > 0) {
    uint64_t ChunkBytes = 8;
    while (ChunkBytes > RemainingBytes) {
      ChunkBytes /= 2;
    }
    Chunks.push_back({Offset, static_cast<unsigned>(ChunkBytes * 8)});
    Offset += ChunkBytes;
    RemainingBytes -= ChunkBytes;
  }
  return Chunks;
}

llvm::Value *getChunkPtr(llvm::IRBuilder<> &B, llvm::Value *BasePtr,
                         const MemoryChunk &C, llvm::Type *ChunkTy) {
  if (C.OffsetInBytes == 0) {
    return BasePtr;
  }
  const uint64_t ChunkBytes = C.Bits / 8;
  assert(C.OffsetInBytes % ChunkBytes == 0 && "Chunk offset not aligned");
  return B.CreateConstInBoundsGEP1_64(ChunkTy, BasePtr,
                                      C.OffsetInBytes / ChunkBytes);
}

llvm::Align getChunkAlign(llvm::Align BaseAlign, const MemoryChunk &C) {
  if (C.OffsetInBytes == 0) {
    return BaseAlign;
  }
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

// Every illegal type the pass can rewrite must be a scalar integer that fits
// into a legal type. Illegal integers nested in vectors or wider than 64 bits
// are not supported by the promotion logic.
bool isPromotableIllegalTy(const llvm::Type *T) {
  return isIllegalIntTy(T) && getPromotedWidth(T->getIntegerBitWidth()) != 0;
}

bool hasIllegalOperand(const llvm::Instruction *I) {
  for (const llvm::Use &U : I->operands()) {
    if (isIllegalIntTy(U.get()->getType())) {
      return true;
    }
  }
  return false;
}

class IntWidthLegalizer {
public:
  IntWidthLegalizer(llvm::Function &F)
      : F{F}, DL{F.getParent()->getDataLayout()} {}

  // Returns true if the function was modified.
  bool run();

private:
  // Replaces `store iN (load iN ptr), ptr` copies of values too wide to be
  // promoted by a sequence of chunk sized load/store pairs.
  bool splitWideCopies();

  bool collectWork();
  bool rewrite();

  // Value of promoted type whose low `N` bits match the low `N` bits of 'V'.
  // Bits above `N` are undefined.
  llvm::Value *getPromoted(llvm::Value *V);

  llvm::Value *maskToWidth(llvm::IRBuilder<> &B, llvm::Value *PromotedV,
                           unsigned Width);
  llvm::Value *signExtendWithin(llvm::IRBuilder<> &B, llvm::Value *PromotedV,
                                unsigned Width);
  // Adjusts a promoted value to 'TargetTy', preserving its low bits.
  llvm::Value *adjustWidth(llvm::IRBuilder<> &B, llvm::Value *V,
                           llvm::Type *TargetTy);

  llvm::Value *emitSplitLoad(llvm::IRBuilder<> &B, llvm::LoadInst *LI,
                             llvm::Type *CombineTy);
  void emitSplitStore(llvm::IRBuilder<> &B, llvm::StoreInst *SI,
                      llvm::Value *PromotedVal);

  llvm::Value *rewriteInstruction(llvm::Instruction *I);

  // Makes sure that phi nodes created by the pass are well formed, even if
  // legalization was aborted half way through.
  void completeCreatedPhis();

  llvm::Function &F;
  const llvm::DataLayout &DL;

  llvm::SmallVector<llvm::Instruction *, 32> Work;
  llvm::SmallVector<llvm::AllocaInst *, 8> IllegalAllocas;
  llvm::DenseMap<llvm::Value *, llvm::Value *> Promoted;
  llvm::DenseMap<llvm::PHINode *, llvm::PHINode *> PhiMap;
};

bool IntWidthLegalizer::splitWideCopies() {
  llvm::SmallVector<llvm::StoreInst *, 8> WideStores;
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I);
      if (!SI || !SI->isSimple()) {
        continue;
      }
      llvm::Type *ValTy = SI->getValueOperand()->getType();
      if (isIllegalIntTy(ValTy) &&
          getPromotedWidth(ValTy->getIntegerBitWidth()) == 0) {
        WideStores.push_back(SI);
      }
    }
  }

  bool Changed = false;
  for (llvm::StoreInst *SI : WideStores) {
    llvm::Value *Val = SI->getValueOperand();
    const unsigned BitWidth = Val->getType()->getIntegerBitWidth();
    auto Chunks = computeChunks(BitWidth);

    llvm::IRBuilder<> B{SI};
    auto *LI = llvm::dyn_cast<llvm::LoadInst>(Val);
    auto *CI = llvm::dyn_cast<llvm::ConstantInt>(Val);

    if (LI && LI->isSimple() && LI->getType() == Val->getType()) {
      // Pure copy: split into chunk sized load/store pairs.
      for (const MemoryChunk &C : Chunks) {
        auto *ChunkTy = llvm::IntegerType::get(F.getContext(), C.Bits);
        llvm::Value *SrcPtr =
            getChunkPtr(B, LI->getPointerOperand(), C, ChunkTy);
        llvm::Value *DstPtr =
            getChunkPtr(B, SI->getPointerOperand(), C, ChunkTy);
        llvm::Value *ChunkVal = B.CreateAlignedLoad(
            ChunkTy, SrcPtr, getChunkAlign(LI->getAlign(), C));
        B.CreateAlignedStore(ChunkVal, DstPtr,
                             getChunkAlign(SI->getAlign(), C));
      }
      SI->eraseFromParent();
      if (LI->use_empty())
        LI->eraseFromParent();
      Changed = true;
    } else if (CI) {
      for (const MemoryChunk &C : Chunks) {
        auto *ChunkTy = llvm::IntegerType::get(F.getContext(), C.Bits);
        llvm::Value *DstPtr =
            getChunkPtr(B, SI->getPointerOperand(), C, ChunkTy);
        llvm::APInt ChunkVal =
            CI->getValue().lshr(C.OffsetInBytes * 8).trunc(C.Bits);
        B.CreateAlignedStore(llvm::ConstantInt::get(ChunkTy, ChunkVal), DstPtr,
                             getChunkAlign(SI->getAlign(), C));
      }
      SI->eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

bool IntWidthLegalizer::collectWork() {
  llvm::ReversePostOrderTraversal<llvm::Function *> RPO{&F};
  for (llvm::BasicBlock *BB : RPO) {
    for (llvm::Instruction &I : *BB) {
      if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        if (isIllegalIntTy(AI->getAllocatedType())) {
          IllegalAllocas.push_back(AI);
        }
      }

      if (!isIllegalIntTy(I.getType()) && !hasIllegalOperand(&I)) {
        continue;
      }

      if (!isSupportedOpcode(&I)) {
        HIPSYCL_DEBUG_WARNING
            << "LegalizeIntWidthsPass: Unsupported instruction using an "
               "integer type of non-standard bit width in function "
            << F.getName().str() << "; clspv may reject this kernel\n";
        return false;
      }

      // The promotion logic only supports scalar integers that fit into a
      // legal type.
      if (isIllegalIntTy(I.getType()) && !isPromotableIllegalTy(I.getType())) {
        return false;
      }
      for (llvm::Use &U : I.operands()) {
        llvm::Type *OpTy = U.get()->getType();
        if (isIllegalIntTy(OpTy) && !isPromotableIllegalTy(OpTy)) {
          return false;
        }
        if (isIllegalIntTy(OpTy) && !llvm::isa<llvm::Instruction>(U.get()) &&
            !llvm::isa<llvm::ConstantInt>(U.get()) &&
            !llvm::isa<llvm::UndefValue>(U.get())) {
          return false;
        }
      }

      Work.push_back(&I);
    }
  }
  return true;
}

llvm::Value *IntWidthLegalizer::getPromoted(llvm::Value *V) {
  assert(isIllegalIntTy(V->getType()) && "Value is already of legal type");

  auto It = Promoted.find(V);
  if (It != Promoted.end()) {
    return It->second;
  }

  llvm::IntegerType *PromTy = getPromotedTy(V->getType());
  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(V))
    return llvm::ConstantInt::get(PromTy,
                                  CI->getValue().zext(PromTy->getBitWidth()));
  if (llvm::isa<llvm::UndefValue>(V)) {
    return llvm::UndefValue::get(PromTy);
  }

  return nullptr;
}

llvm::Value *IntWidthLegalizer::maskToWidth(llvm::IRBuilder<> &B,
                                            llvm::Value *PromotedV,
                                            unsigned Width) {
  const unsigned PromWidth = PromotedV->getType()->getIntegerBitWidth();
  if (PromWidth == Width) {
    return PromotedV;
  }
  return B.CreateAnd(
      PromotedV,
      llvm::ConstantInt::get(PromotedV->getType(),
                             llvm::APInt::getLowBitsSet(PromWidth, Width)));
}

llvm::Value *IntWidthLegalizer::signExtendWithin(llvm::IRBuilder<> &B,
                                                 llvm::Value *PromotedV,
                                                 unsigned Width) {
  const unsigned PromWidth = PromotedV->getType()->getIntegerBitWidth();
  if (PromWidth == Width) {
    return PromotedV;
  }
  llvm::Value *ShiftAmount =
      llvm::ConstantInt::get(PromotedV->getType(), PromWidth - Width);
  return B.CreateAShr(B.CreateShl(PromotedV, ShiftAmount), ShiftAmount);
}

llvm::Value *IntWidthLegalizer::adjustWidth(llvm::IRBuilder<> &B,
                                            llvm::Value *V,
                                            llvm::Type *TargetTy) {
  const unsigned FromWidth = V->getType()->getIntegerBitWidth();
  const unsigned ToWidth = TargetTy->getIntegerBitWidth();
  if (FromWidth == ToWidth) {
    return V;
  }
  if (FromWidth > ToWidth) {
    return B.CreateTrunc(V, TargetTy);
  }
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
    if (C.OffsetInBytes != 0) {
      Extended = B.CreateShl(
          Extended, llvm::ConstantInt::get(CombineTy, C.OffsetInBytes * 8));
    }
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
    if (C.OffsetInBytes != 0) {
      ChunkVal =
          B.CreateLShr(ChunkVal, llvm::ConstantInt::get(Val->getType(),
                                                        C.OffsetInBytes * 8));
    }
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
    if (!Val) {
      return nullptr;
    }
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
    if (!T || !Fa) {
      return nullptr;
    }
    return B.CreateSelect(I->getOperand(0), T, Fa);
  }
  case llvm::Instruction::ICmp: {
    auto *Cmp = llvm::cast<llvm::ICmpInst>(I);
    const unsigned Width = Cmp->getOperand(0)->getType()->getIntegerBitWidth();
    llvm::Value *L = promotedOperand(0);
    llvm::Value *R = promotedOperand(1);
    if (!L || !R) {
      return nullptr;
    }
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
    if (!Src) {
      return nullptr;
    }
    llvm::Type *SrcTy = I->getOperand(0)->getType();
    llvm::Type *DstTy = I->getType();
    llvm::Type *NewDstTy = isIllegalIntTy(DstTy) ? getPromotedTy(DstTy) : DstTy;

    if (isIllegalIntTy(SrcTy)) {
      const unsigned SrcWidth = SrcTy->getIntegerBitWidth();
      if (I->getOpcode() == llvm::Instruction::ZExt) {
        Src = maskToWidth(B, Src, SrcWidth);
      } else if (I->getOpcode() == llvm::Instruction::SExt) {
        Src = signExtendWithin(B, Src, SrcWidth);
      }
    }

    const unsigned SrcPromWidth = Src->getType()->getIntegerBitWidth();
    const unsigned DstPromWidth = NewDstTy->getIntegerBitWidth();
    if (SrcPromWidth == DstPromWidth) {
      return Src;
    }
    if (SrcPromWidth > DstPromWidth) {
      return B.CreateTrunc(Src, NewDstTy);
    }
    // Widening a legal source keeps the original signedness, widening an
    // already sign/zero extended illegal source can use either.
    if (I->getOpcode() == llvm::Instruction::SExt) {
      return B.CreateSExt(Src, NewDstTy);
    }
    return B.CreateZExt(Src, NewDstTy);
  }
  default:
    break;
  }

  auto *BinOp = llvm::dyn_cast<llvm::BinaryOperator>(I);
  if (!BinOp) {
    return nullptr;
  }

  const unsigned Width = BinOp->getType()->getIntegerBitWidth();
  llvm::Value *L = promotedOperand(0);
  llvm::Value *R = promotedOperand(1);
  if (!L || !R) {
    return nullptr;
  }

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

bool IntWidthLegalizer::rewrite() {
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
    if (isIllegalIntTy(I->getType())) {
      Promoted[I] = New;
    } else {
      I->replaceAllUsesWith(New);
    }
  }

  // Now that all values have a legalized counterpart, patch the phi nodes.
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

  // Erase the original instructions. Their remaining uses are all among the
  // instructions being erased.
  for (llvm::Instruction *I : llvm::reverse(Work)) {
    I->dropAllReferences();
  }
  for (llvm::Instruction *I : llvm::reverse(Work)) {
    I->eraseFromParent();
  }

  return true;
}

bool IntWidthLegalizer::run() {
  bool Changed = splitWideCopies();

  for (const llvm::Argument &A : F.args()) {
    if (isIllegalIntTy(A.getType())) {
      return Changed;
    }
  }
  if (isIllegalIntTy(F.getReturnType())) {
    return Changed;
  }

  if (!collectWork()) {
    return Changed;
  }

  if (!Work.empty()) {
    if (!rewrite()) {
      HIPSYCL_DEBUG_WARNING
          << "LegalizeIntWidthsPass: Failed to legalize integer widths in "
             "function "
          << F.getName().str() << "\n";
    }
    Changed = true;
  }

  // Allocas of illegal types are replaced by equally sized byte arrays; all
  // accesses to them have been split into legal accesses above.
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

// Reports integer types of non-standard width that survived legalization, as
// those would result in an obscure clspv error message later on.
void warnOnRemainingIllegalTypes(llvm::Function &F) {
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      bool Illegal = isIllegalIntTy(I.getType());
      for (llvm::Use &U : I.operands()) {
        Illegal |= isIllegalIntTy(U.get()->getType());
      }
      if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        Illegal |= isIllegalIntTy(AI->getAllocatedType());
      }

      if (Illegal) {
        HIPSYCL_DEBUG_WARNING
            << "LegalizeIntWidthsPass: Function " << F.getName().str()
            << " still contains integer types of non-standard bit width, "
               "which clspv may not be able to lower to SPIR-V\n";
        return;
      }
    }
  }
}

} // namespace

llvm::PreservedAnalyses
LegalizeIntWidthsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
  if (F.isDeclaration()) {
    return llvm::PreservedAnalyses::all();
  }

  IntWidthLegalizer Legalizer{F};
  const bool Changed = Legalizer.run();

  warnOnRemainingIllegalTypes(F);

  return Changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
