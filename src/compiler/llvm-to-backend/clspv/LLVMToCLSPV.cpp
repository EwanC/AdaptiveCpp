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
#include "hipSYCL/compiler/llvm-to-backend/clspv/LLVMToCLSPV.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"
#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/compiler/llvm-to-backend/Utils.hpp"
#include "hipSYCL/compiler/llvm-to-backend/clspv/AddrSpaceCastRemovalPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/clspv/ConstantAddrSpacePass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/clspv/FoldChainedGEPsPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/clspv/MemsetLoweringPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/clspv/RemoveUnusedIntrinsicsPass.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <signal.h>
#include <sys/wait.h>
#endif

namespace hipsycl {
namespace compiler {

namespace {

static const char *DynamicLocalMemArrayName =
    "__acpp_sscp_spirv_dynamic_local_mem";

bool setDynamicLocalMemoryCapacity(llvm::Module &M, unsigned numBytes) {
  llvm::GlobalVariable *GV = M.getGlobalVariable(DynamicLocalMemArrayName);

  if (!GV) {
    // If non-zero number of bytes are needed, not finding the global variable
    // is an error.
    return numBytes == 0;
  }

  const unsigned AddressSpace = GV->getAddressSpace();
  if (numBytes > 0) {
    unsigned numInts = (numBytes + 4 - 1) / 4;
    llvm::Type *T =
        llvm::ArrayType::get(llvm::Type::getInt32Ty(M.getContext()), numInts);

    // Must initialized to undef to be valid SPIR-V
    llvm::GlobalVariable *NewVar = new llvm::GlobalVariable(
        M, T, false, llvm::GlobalValue::InternalLinkage,
        llvm::UndefValue::get(T), GV->getName() + ".resized", nullptr,
        llvm::GlobalVariable::ThreadLocalMode::NotThreadLocal, AddressSpace);

    NewVar->setAlignment(GV->getAlign());
    llvm::Value *V = llvm::ConstantExpr::getPointerCast(NewVar, GV->getType());
    GV->replaceAllUsesWith(V);
    GV->eraseFromParent();
  } else {
    // If no bytes of local memory used but Global memory exists, then set
    // global variable to a nullptr.
    llvm::Type *T = llvm::PointerType::get(M.getContext(), AddressSpace);
    llvm::GlobalVariable *NewVar = new llvm::GlobalVariable(
        M, T, false, llvm::GlobalValue::InternalLinkage,
        llvm::Constant::getNullValue(T), GV->getName() + ".null", nullptr,
        llvm::GlobalVariable::ThreadLocalMode::NotThreadLocal, AddressSpace);
    GV->replaceAllUsesWith(NewVar);
    GV->eraseFromParent();
  }
  return true;
}

void assignSPIRCallConvention(llvm::Function *F) {
  if (F->getCallingConv() != llvm::CallingConv::SPIR_FUNC)
    F->setCallingConv(llvm::CallingConv::SPIR_FUNC);

  // All callers must use spir_func calling convention
  for (auto U : F->users()) {
    if (auto CI = llvm::dyn_cast<llvm::CallBase>(U)) {
      CI->setCallingConv(llvm::CallingConv::SPIR_FUNC);
    }
  }
}

void rewriteZeroSizeArrayGEPs(llvm::Module &M) {
  llvm::SmallVector<llvm::GetElementPtrInst *> GEPs;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *GEPInst = llvm::dyn_cast<llvm::GetElementPtrInst>(&I)) {
          auto *SourceTy = GEPInst->getSourceElementType();

          if (GEPInst->getNumIndices() > 0 && SourceTy->isArrayTy()) {
            llvm::Value *FirstIndex = GEPInst->idx_begin()->get();
            int64_t FirstIndexVal = -1;
            if (llvm::ConstantInt *CI =
                    llvm::dyn_cast<llvm::ConstantInt>(FirstIndex)) {
              if (CI->getBitWidth() <= 64) {
                FirstIndexVal = CI->getSExtValue();
              }
            }

            if (FirstIndexVal == 0 && SourceTy->getArrayNumElements() == 0) {
              GEPs.push_back(GEPInst);
            }
          }
        }
      }
    }
  }

  for (auto *GEPInst : GEPs) {
    auto *ReplacementType = llvm::ArrayType::get(
        GEPInst->getSourceElementType()->getArrayElementType(), 1);

    llvm::SmallVector<llvm::Value *> Indices;
    for (llvm::Use &Op : GEPInst->indices()) {
      Indices.push_back(Op.get());
    }

    llvm::Value *PointerOperand = GEPInst->getPointerOperand();
    // Old LLVM needs bitcast
#if LLVM_VERSION_MAJOR < 17
    llvm::Type *BitcastTarget = llvm::PointerType::get(
        ReplacementType, GEPInst->getPointerAddressSpace());
    PointerOperand =
        new llvm::BitCastInst(PointerOperand, BitcastTarget, "", GEPInst);
#endif
    llvm::GetElementPtrInst *NewGEP = llvm::GetElementPtrInst::Create(
        ReplacementType, PointerOperand, Indices, "",
        llvmutils::makeInsertionPoint(GEPInst));
    if (GEPInst->isInBounds())
      NewGEP->setIsInBounds(true);
    GEPInst->replaceAllUsesWith(NewGEP);
    GEPInst->dropAllReferences();
    GEPInst->eraseFromParent();
  }
}

bool isKernelDefinition(const llvm::Function &F) {
  return !F.isDeclaration() &&
         F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL;
}

// Removes the functions and global variables that are no longer used, e.g.
// after kernels were removed from the module.
void removeUnusedGlobals(llvm::Module &M) {
  // GlobalDCEPass does not use any analyses
  llvm::ModuleAnalysisManager MAM;
  llvm::GlobalDCEPass{}.run(M, MAM);
}

// Creates a copy of M that only contains the given kernel and the functions
// and global variables it uses.
std::unique_ptr<llvm::Module> cloneKernel(const llvm::Module &M,
                                          llvm::StringRef Kernel) {
  // The other kernels are created as declarations without uses, which are
  // removed together with everything only they use.
  llvm::ValueToValueMapTy VMap;
  std::unique_ptr<llvm::Module> Clone =
      llvm::CloneModule(M, VMap, [&](const llvm::GlobalValue *GV) {
        auto *F = llvm::dyn_cast<llvm::Function>(GV);
        return !F || !isKernelDefinition(*F) || F->getName() == Kernel;
      });
  removeUnusedGlobals(*Clone);
  return Clone;
}

// A clspv process compiling a single kernel, of which the output is discarded
struct ClspvJob {
  std::string Kernel;
  std::string InputFile;
  std::string OutputFile;
  llvm::sys::ProcessInfo Process;
  std::chrono::steady_clock::time_point StartTime;
};

void removeJobFiles(const ClspvJob &Job) {
  (void)llvm::sys::fs::remove(Job.InputFile);
  (void)llvm::sys::fs::remove(Job.OutputFile);
}

// Writes M to a temporary file and starts clspv on it, without waiting for
// clspv to finish.
bool startClspvJob(const llvm::Module &M, const std::vector<std::string> &Args,
                   ClspvJob &Job) {
  int InputFD = 0;
  llvm::SmallString<128> InputFile;
  if (llvm::sys::fs::createTemporaryFile("acpp-sscp-clspv", "bc", InputFD,
                                         InputFile))
    return false;
  Job.InputFile = InputFile.str().str();
  {
    llvm::raw_fd_ostream InputStream{InputFD, true};
    llvm::WriteBitcodeToFile(M, InputStream);
  }

  llvm::SmallString<128> OutputFile;
  if (llvm::sys::fs::createTemporaryFile("acpp-sscp-clspv", "spv", OutputFile))
    return false;
  Job.OutputFile = OutputFile.str().str();

  std::string CLSPV = ACPP_CLSPV_PATH;
  std::string OutputArg = "-o=" + Job.OutputFile;
  llvm::SmallVector<llvm::StringRef, 16> Invocation{CLSPV};
  for (const auto &A : Args)
    Invocation.push_back(A);
  Invocation.push_back(OutputArg);
  Invocation.push_back(Job.InputFile);
  HIPSYCL_DEBUG_INFO << "LLVMToCLSPV: Invoking " << llvm::join(Invocation, " ")
                     << "\n";

  // Discard the output of clspv, the kernels it fails on are reported instead
  std::optional<llvm::StringRef> Redirects[] = {
      llvm::StringRef{}, llvm::StringRef{}, llvm::StringRef{}};
  bool ExecutionFailed = false;
  Job.Process = llvm::sys::ExecuteNoWait(
      CLSPV, Invocation, std::nullopt, Redirects, 0, nullptr, &ExecutionFailed);
  Job.StartTime = std::chrono::steady_clock::now();
  return !ExecutionFailed;
}

// Returns the exit code of the process if it has finished, without blocking.
std::optional<int> pollProcess(const llvm::sys::ProcessInfo &Process) {
#ifdef _WIN32
  llvm::sys::ProcessInfo Result = llvm::sys::Wait(Process, 0u);
  if (Result.Pid == 0)
    return std::nullopt;
  return Result.ReturnCode;
#else
  // Not llvm::sys::Wait, as it replaces the SIGALRM handler of the
  // application, even for a non-blocking wait.
  int Status = 0;
  pid_t Pid = ::waitpid(Process.Pid, &Status, WNOHANG);
  if (Pid == 0 || (Pid == -1 && errno == EINTR))
    return std::nullopt;
  if (Pid == -1)
    return -1;
  return WIFEXITED(Status) ? WEXITSTATUS(Status) : -2;
#endif
}

void killProcess(const llvm::sys::ProcessInfo &Process) {
#ifdef _WIN32
  // Terminates the process if it does not finish within a second
  llvm::sys::Wait(Process, 1u);
#else
  ::kill(Process.Pid, SIGKILL);
  int Status = 0;
  while (::waitpid(Process.Pid, &Status, 0) == -1 && errno == EINTR) {
  }
#endif
}

} // namespace

LLVMToCLSPVTranslator::LLVMToCLSPVTranslator(const std::vector<std::string> &KN)
    : LLVMToBackendTranslator{static_cast<int>(sycl::AdaptiveCpp_jit::
                                                   compiler_backend::clspv),
                              KN, KN},
      KernelNames{KN} {}

bool LLVMToCLSPVTranslator::toBackendFlavor(llvm::Module &M, PassHandler &PH) {

#if LLVM_VERSION_MAJOR > 20
  M.setTargetTriple(llvm::Triple("spirv64-unknown-vulkan"));
#else
  M.setTargetTriple("spirv64-unknown-vulkan");
#endif
  M.setDataLayout("e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:"
                  "256-v512:512-v1024:1024-n8:16:32:64-G1");

  rewriteZeroSizeArrayGEPs(M);

  AddressSpaceMap ASMap = getAddressSpaceMap();
  KernelFunctionParameterRewriter ParamRewriter{
      // ByVal attribute for all aggregates passed in by-value
      KernelFunctionParameterRewriter::ByValueArgAttribute::ByVal,
      // Those pointers to by-value data should be in private AS
      ASMap[AddressSpace::Private],
      // Actual pointers should be in global memory
      ASMap[AddressSpace::Global],
      // We don't need to wrap pointer types
      false};

  ParamRewriter.run(M, KernelNames, *PH.ModuleAnalysisManager);

  for (auto KernelName : KernelNames) {
    HIPSYCL_DEBUG_INFO << "LLVMToCLSPV: Setting up kernel " << KernelName
                       << "\n";
    if (auto *F = M.getFunction(KernelName)) {
      applyKernelProperties(F);
    }
  }

  for (auto &F : M) {
    if (F.getCallingConv() != llvm::CallingConv::SPIR_KERNEL) {
      // All functions must be marked as spir_func
      assignSPIRCallConvention(&F);
    }
  }

  std::string BuiltinBitcodeFile = common::filesystem::join_path(
      getBitcodePath(), "libkernel-sscp-clspv-full.bc");

#if LLVM_VERSION_MAJOR > 20
  if (!this->linkBitcodeFile(M, BuiltinBitcodeFile, M.getTargetTriple().str(),
                             M.getDataLayoutStr()))
#else
  if (!this->linkBitcodeFile(M, BuiltinBitcodeFile, M.getTargetTriple(),
                             M.getDataLayoutStr()))
#endif
    return false;

  HIPSYCL_DEBUG_INFO << "LLVMToCLSPV: Configuring kernel for "
                     << DynamicLocalMemSize << " bytes of local memory\n";
  if (!setDynamicLocalMemoryCapacity(M, DynamicLocalMemSize)) {
    HIPSYCL_DEBUG_WARNING << "Could not set dynamic local memory size; this "
                             "could imply that local memory "
                             "requested by the application is not actually "
                             "used inside kernels\n";
  }

  // kernel debugging as future work
  llvm::StripDebugInfo(M);

  // Change the behavior of the 'wchar_size' metadata node from 'error'
  // to 'override'. This is due to discrepancy on Windows where we set
  // this to '2' but clspv links in libclc builtins with metadata set
  // to '4'.
  llvm::NamedMDNode *ModFlags = M.getOrInsertModuleFlagsMetadata();
  for (unsigned i = 0; i < ModFlags->getNumOperands(); ++i) {
    llvm::MDNode *Flag = ModFlags->getOperand(i);
    if (llvm::cast<llvm::MDString>(Flag->getOperand(1))->getString() ==
        "wchar_size") {
      auto &Context = M.getContext();
      llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Context);
      llvm::Metadata *Ops[3] = {
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
              Int32Ty, llvm::Module::ModFlagBehavior::Override)),
          llvm::MDString::get(Context, "wchar_size"), Flag->getOperand(2)};
      ModFlags->setOperand(i, llvm::MDNode::get(Context, Ops));
      break;
    }
  }

  return true;
}

std::vector<std::string> LLVMToCLSPVTranslator::getClspvArgs() const {
  std::vector<std::string> Args{"-x=ir", "--physical-storage-buffers",
                                "-spv-version=1.5", "--arch=spir64",
                                "-long-vector"};

  // Workaround for the Qualcomm Adreno driver on Android that struggles to
  // consume SPIR-V where the physical pointer arguments are passed in a push
  // constant with non zero member index. To avoid this pass each argument as
  // its own uniform buffer.
  if (DeviceName.find("Adreno") != std::string::npos) {
    Args.push_back("--pod-ubo");
    Args.push_back("--cluster-pod-kernel-args=0");
  }

  if (!MaxPushConstantSize.empty()) {
    Args.push_back("-max-pushconstant-size=" + MaxPushConstantSize);
  }

  if (!MaxUniformBufferRange.empty()) {
    Args.push_back("-max-ubo-size=" + MaxUniformBufferRange);
  }

  return Args;
}

bool LLVMToCLSPVTranslator::runClspv(const llvm::Module &M, std::string &Out,
                                     int &ExitCode) {
  bool Create = true;
  auto consumeError = [&](std::error_code EC) {
    if (EC) {
      if (Create)
        this->registerError("LLVMToCLSPV: Could not create temp file: " +
                            EC.message());
      else
        this->registerError("LLVMToCLSPV: Could not remove temp file: " +
                            EC.message());
      return false;
    }
    return true;
  };

  int InputFD = 0;
  llvm::SmallVector<char> InputFileNameBuf;
  // don't use fs::TempFile, as we can't unlock the file for the clang
  // invocation later... (Windows)
  if (!consumeError(llvm::sys::fs::createTemporaryFile(
          "acpp-sscp-clspv", "bc", InputFD, InputFileNameBuf,
          llvm::sys::fs::OF_None)))
    return false;
  std::string InputFileName = InputFileNameBuf.data();

  llvm::SmallVector<char> OutputFile;
  if (!consumeError(llvm::sys::fs::createTemporaryFile(
          "acpp-sscp-clspv", "spv", OutputFile, llvm::sys::fs::OF_None)))
    return false;
  std::string OutputFileName = OutputFile.data();

  Create = false;
  AtScopeExit DestroyInputFile(
      [&]() { consumeError(llvm::sys::fs::remove(InputFileName)); });
  AtScopeExit DestroyOutputFile(
      [&]() { consumeError(llvm::sys::fs::remove(OutputFileName)); });

  {
    llvm::raw_fd_ostream InputStream{InputFD, true};

    llvm::WriteBitcodeToFile(M, InputStream);
    InputStream.flush();
  }

  std::string CLSPV = ACPP_CLSPV_PATH;

  std::vector<std::string> Args = getClspvArgs();
  Args.push_back("-o=" + OutputFileName);

  llvm::SmallVector<llvm::StringRef, 16> Invocation{CLSPV};
  for (const auto &A : Args)
    Invocation.push_back(A);
  Invocation.push_back(InputFileName);

  std::string ArgString;
  for (const auto &S : Invocation) {
    ArgString += S;
    ArgString += " ";
  }
  HIPSYCL_DEBUG_INFO << "LLVMToCLSPV: Invoking " << ArgString << "\n";

  ExitCode = executeAndWait(CLSPV, Invocation);
  if (ExitCode != 0)
    return true;

  auto ReadResult = llvm::MemoryBuffer::getFile(OutputFileName);

  if (auto Err = ReadResult.getError()) {
    this->registerError("LLVMToCLSPV: Could not read result file" +
                        Err.message());
    return false;
  }

  Out = ReadResult->get()->getBuffer();

  return true;
}

bool LLVMToCLSPVTranslator::removeKernelsClspvFailsOn(llvm::Module &M) {
  std::vector<std::string> Kernels;
  for (auto &F : M)
    if (isKernelDefinition(F))
      Kernels.push_back(F.getName().str());
  if (Kernels.size() < 2)
    return false;

  HIPSYCL_DEBUG_WARNING << "LLVMToCLSPV: clspv failed to compile a module with "
                        << Kernels.size()
                        << " kernels, compiling the kernels separately to "
                           "find the ones it fails on\n";

  const std::vector<std::string> Args = getClspvArgs();
  // A kernel clspv hangs on is treated like a kernel it fails on. 0 disables
  // the timeout.
  const std::chrono::seconds Timeout{getEnvironmentVariableOrDefault<unsigned>(
      "clspv_isolation_timeout", 120)};
  const std::size_t MaxProcesses =
      std::max(1u, std::thread::hardware_concurrency());

  // Pairs of kernel name and the reason why it is removed
  std::vector<std::pair<std::string, std::string>> FailingKernels;
  std::vector<ClspvJob> Jobs;
  std::size_t NumStarted = 0;
  while (NumStarted < Kernels.size() || !Jobs.empty()) {
    while (NumStarted < Kernels.size() && Jobs.size() < MaxProcesses) {
      ClspvJob Job;
      Job.Kernel = Kernels[NumStarted++];
      if (startClspvJob(*cloneKernel(M, Job.Kernel), Args, Job))
        Jobs.push_back(std::move(Job));
      else
        removeJobFiles(Job);
    }

    for (auto It = Jobs.begin(); It != Jobs.end();) {
      std::string FailureReason;
      if (std::optional<int> R = pollProcess(It->Process)) {
        if (*R != 0)
          FailureReason =
              "clspv failed to compile it with exit code " + std::to_string(*R);
      } else if (Timeout.count() > 0 &&
                 std::chrono::steady_clock::now() - It->StartTime > Timeout) {
        killProcess(It->Process);
        FailureReason = "clspv did not finish compiling it within " +
                        std::to_string(Timeout.count()) + " seconds";
      } else {
        ++It;
        continue;
      }

      if (!FailureReason.empty())
        FailingKernels.emplace_back(It->Kernel, FailureReason);
      removeJobFiles(*It);
      It = Jobs.erase(It);
    }

    if (!Jobs.empty())
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
  }

  if (FailingKernels.empty() || FailingKernels.size() == Kernels.size())
    return false;

  for (const auto &[Kernel, Reason] : FailingKernels) {
    HIPSYCL_DEBUG_WARNING << "LLVMToCLSPV: Removing kernel " << Kernel
                          << " from the module, as " << Reason << "\n";
    llvm::Function *F = M.getFunction(Kernel);
    if (!F->use_empty())
      F->replaceAllUsesWith(llvm::PoisonValue::get(F->getType()));
    F->eraseFromParent();
  }
  removeUnusedGlobals(M);
  return true;
}

bool LLVMToCLSPVTranslator::translateToBackendFormat(
    llvm::Module &FlavoredModule, std::string &out) {
  int R = 0;
  if (!runClspv(FlavoredModule, out, R))
    return false;

  // At ACPP_ADAPTIVITY_LEVEL=0 all kernels of a translation unit are compiled
  // into one module, including kernels the application never launches on the
  // Vulkan backend. A kernel clspv fails on must not break the other kernels.
  if (R != 0 && removeKernelsClspvFailsOn(FlavoredModule) &&
      !runClspv(FlavoredModule, out, R))
    return false;

  if (R != 0) {
    this->registerError("LLVMToCLSPV: clspv invocation failed with exit code " +
                        std::to_string(R));
    return false;
  }

  return true;
}

bool LLVMToCLSPVTranslator::applyBuildOption(const std::string &Option,
                                             const std::string &Value) {
  if (Option == "spirv-dynamic-local-mem-allocation-size") {
    this->DynamicLocalMemSize = static_cast<unsigned>(std::stoi(Value));
    return true;
  }

  if (Option == "-max-pushconstant-size") {
    this->MaxPushConstantSize = Value;
    return true;
  }

  if (Option == "-max-ubo-size") {
    this->MaxUniformBufferRange = Value;
    return true;
  }

  if (Option == "-device-name") {
    this->DeviceName = Value;
    return true;
  }

  return false;
}

bool LLVMToCLSPVTranslator::applyBuildFlag(const std::string &Flag) {
  return false;
}

bool LLVMToCLSPVTranslator::isKernelAfterFlavoring(llvm::Function &F) {
  return F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL;
}

AddressSpaceMap LLVMToCLSPVTranslator::getAddressSpaceMap() const {
  AddressSpaceMap ASMap;

  ASMap[AddressSpace::Private] = 0;
  ASMap[AddressSpace::Global] = 1;
  ASMap[AddressSpace::Constant] = 2;
  ASMap[AddressSpace::Local] = 3;
  ASMap[AddressSpace::Generic] = 4;
  ASMap[AddressSpace::AllocaDefault] = 4;
  ASMap[AddressSpace::GlobalVariableDefault] = 1;
  ASMap[AddressSpace::ConstantGlobalVariableDefault] = 1;

  return ASMap;
}

bool LLVMToCLSPVTranslator::optimizeFlavoredIR(llvm::Module &M,
                                               PassHandler &PH) {
  bool Result = LLVMToBackendTranslator::optimizeFlavoredIR(M, PH);
  if (!Result)
    return false;

  // Transformation passes to make the LLVM-IR consumable by clspv
  llvm::ModulePassManager MPM;
  MPM.addPass(
      llvm::createModuleToFunctionPassAdaptor(RemoveUnusedIntrinsicsPass()));
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(MemsetLoweringPass()));
  MPM.addPass(
      llvm::createModuleToFunctionPassAdaptor(FoldChainedGEPsPass()));
  MPM.addPass(
      llvm::createModuleToFunctionPassAdaptor(AddrSpaceCastRemovalPass()));
  MPM.addPass(ConstantAddrSpacePass());
  MPM.run(M, *PH.ModuleAnalysisManager);

  return Result;
}

void LLVMToCLSPVTranslator::migrateKernelProperties(llvm::Function *From,
                                                    llvm::Function *To) {
  removeKernelProperties(From);
  applyKernelProperties(To);
}

void LLVMToCLSPVTranslator::applyKernelProperties(llvm::Function *F) {
  F->setCallingConv(llvm::CallingConv::SPIR_KERNEL);

  llvm::Module &M = *F->getParent();

  if (KnownGroupSizeX != 0 && KnownGroupSizeY != 0 && KnownGroupSizeZ != 0) {
    llvm::SmallVector<llvm::Metadata *> MDs;
    MDs.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
        llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeX)));
    MDs.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
        llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeY)));
    MDs.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
        llvm::Type::getInt32Ty(M.getContext()), KnownGroupSizeZ)));

    static const char *ReqdWGSize = "reqd_work_group_size";
    F->setMetadata(ReqdWGSize, llvm::MDNode::get(M.getContext(), MDs));
  }
}

void LLVMToCLSPVTranslator::removeKernelProperties(llvm::Function *F) {
  assignSPIRCallConvention(F);
  for (int i = 0; i < F->getFunctionType()->getNumParams(); ++i) {
    if (F->getArg(i)->hasAttribute(llvm::Attribute::ByVal)) {
      F->getArg(i)->removeAttr(llvm::Attribute::ByVal);
    }
  }
  F->clearMetadata();
}

std::unique_ptr<LLVMToBackendTranslator>
createLLVMToCLSPVTranslator(const std::vector<std::string> &KernelNames) {
  return std::make_unique<LLVMToCLSPVTranslator>(KernelNames);
}

} // namespace compiler
} // namespace hipsycl
