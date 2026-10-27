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

#include "../LLVMToBackend.hpp"

#include <string>
#include <vector>

namespace hipsycl {
namespace compiler {

class LLVMToCLSPVTranslator : public LLVMToBackendTranslator {
public:
  LLVMToCLSPVTranslator(const std::vector<std::string> &KernelNames);

  virtual ~LLVMToCLSPVTranslator() {}

  bool prepareBackendFlavor(llvm::Module &M) override { return true; }
  bool toBackendFlavor(llvm::Module &M, PassHandler &PH) override;
  bool translateToBackendFormat(llvm::Module &FlavoredModule,
                                std::string &Out) override;

protected:
  bool applyBuildOption(const std::string &Option,
                        const std::string &Value) override;
  bool applyBuildFlag(const std::string &Flag) override;
  bool isKernelAfterFlavoring(llvm::Function &F) override;
  AddressSpaceMap getAddressSpaceMap() const override;
  bool optimizeFlavoredIR(llvm::Module &M, PassHandler &PH) override;
  void migrateKernelProperties(llvm::Function *From,
                               llvm::Function *To) override;

private:
  void applyKernelProperties(llvm::Function *F);
  void removeKernelProperties(llvm::Function *F);

  // Options passed to clspv, apart from the input and output files
  std::vector<std::string> getClspvArgs() const;
  // Compiles M with clspv, and on success stores the SPIR-V binary in Out.
  // ExitCode is set to the exit code of clspv. Returns false if an error
  // other than clspv failing occurred, e.g. if temporary files could not be
  // created.
  bool runClspv(const llvm::Module &M, std::string &Out, int &ExitCode);
  // Compiles each kernel of M separately with clspv, and removes the kernels
  // clspv fails on from M. Returns true if kernels were removed, and other
  // kernels remain in M.
  bool removeKernelsClspvFailsOn(llvm::Module &M);

  std::vector<std::string> KernelNames;
  unsigned DynamicLocalMemSize = 0;
  std::string MaxPushConstantSize;
  std::string MaxUniformBufferRange;
  std::string DeviceName;
};

} // namespace compiler
} // namespace hipsycl
