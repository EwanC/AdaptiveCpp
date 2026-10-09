// SPDX-License-Identifier: BSD-2-Clause
#include "hipSYCL/compiler/llvm-to-backend/clspv/LegalizeIntWidthsPass.hpp"
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  llvm::LLVMContext Context;
  llvm::SMDiagnostic Diagnostic;
  auto M = llvm::parseIRFile(argv[1], Diagnostic, Context);
  if (!M) {
    Diagnostic.print(argv[0], llvm::errs());
    return 1;
  }
  if (llvm::verifyModule(*M, &llvm::errs()))
    return 1;
  llvm::FunctionAnalysisManager FAM;
  for (llvm::Function &F : *M)
    hipsycl::compiler::LegalizeIntWidthsPass{}.run(F, FAM);
  if (llvm::verifyModule(*M, &llvm::errs()))
    return 1;
  for (llvm::Function &F : *M)
    hipsycl::compiler::LegalizeIntWidthsPass{}.run(F, FAM);
  if (llvm::verifyModule(*M, &llvm::errs()))
    return 1;
  M->print(llvm::outs(), nullptr);
}
