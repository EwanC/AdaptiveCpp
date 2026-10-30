# BUG-11: clspv aborts on byte-offset local-memory accesses (scan kernels)

| | |
|---|---|
| File at | AdaptiveCpp: <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`); clspv: <https://github.com/google/clspv/issues/new> (second body below) |
| Suggested titles | AdaptiveCpp: `[VK] clspv aborts compiling kernels that index a char local_accessor with an unaligned offset (HeCBench scan)` · clspv: `SimplifyPointerBitcastPass: UNREACHABLE in GetIdxsForTyFromOffset for an i8 GEP at an unaligned offset into an i32 array` |
| Backend / devices | Vulkan (clspv compile step), therefore all Vulkan drivers |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-5 (scan rows, M1 R6 and M2 R14), BUG-11; HeCBench `scan-sycl` |
| Related | [BUG-7](BUG-7-sscp-failed-jit-retried-every-launch.md): after this failure the kernel is recompiled on every launch |
| Before filing with clspv | The clspv used here is upstream `a3d3e261` plus one local commit (`2ecc89b5`, branch `replace_pointer_bitcast_struct_split`) that changes `lib/ReplacePointerBitcastPass.cpp` and `lib/BitcastUtils.h` only. The abort is in `SimplifyPointerBitcastPass.cpp` / `BitcastUtils.cpp`, which that commit does not touch, but re-run the `clspv-opt` reproducer on an unpatched upstream build first. |

---

## Body for the AdaptiveCpp issue

**Bug summary**

On the Vulkan backend, kernels that access a `local_accessor<char>` at an index
of the form `offset * (2 * thid + 1) - 1` (the up-sweep of a work-efficient
scan, as in HeCBench `scan-sycl`) fail to JIT-compile: clspv aborts with
`UNREACHABLE executed at lib/BitcastUtils.cpp:1443` ("Unexpected offset for type
in GetIdxsForTyFromOffset"). The kernel never runs. The same program compiles
and gives the expected result with the OpenCL and CUDA backends.

**To Reproduce**

`repro.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
  sycl::queue q{sycl::property::queue::in_order()};
  const int n = 64;
  char *data = sycl::malloc_device<char>(n, q);
  q.memset(data, 1, n);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<char, 1> temp(n, h);
    h.parallel_for(sycl::nd_range<1>{n / 2, n / 2}, [=](sycl::nd_item<1> it) {
      int thid = it.get_local_id(0);
      temp[2 * thid] = data[2 * thid];
      temp[2 * thid + 1] = data[2 * thid + 1];
      int offset = 1;
      for (int d = n >> 1; d > 0; d >>= 1) { // up-sweep of a work-efficient scan
        sycl::group_barrier(it.get_group());
        if (thid < d) {
          int ai = offset * (2 * thid + 1) - 1;
          int bi = offset * (2 * thid + 2) - 1;
          temp[bi] += temp[ai];
        }
        offset *= 2;
      }
      sycl::group_barrier(it.get_group());
      data[2 * thid] = temp[2 * thid];
      data[2 * thid + 1] = temp[2 * thid + 1];
    });
  });
  q.wait();
  char last = 0;
  q.memcpy(&last, data + n - 1, 1).wait();
  printf("data[n-1] = %d (expected %d)\n", last, n);
  sycl::free(data, q);
}
```

```sh
acpp -O3 --acpp-targets=generic repro.cpp -o repro
ACPP_VISIBILITY_MASK="omp;vk:NVIDIA" ./repro     # same with vk:MTL and vk:llvmpipe
ACPP_VISIBILITY_MASK="omp;ocl:Arc" ./repro       # for comparison
```

Vulkan output:

```
Err: SrcTy = [16 x i32] - DstTy = i8 - Ty = [16 x i32] - CstVal = -8
Unexpected offset for type in GetIdxsForTyFromOffset
UNREACHABLE executed at .../clspv/lib/BitcastUtils.cpp:1443!
[AdaptiveCpp Error] from .../include/hipSYCL/glue/llvm-sscp/jit.hpp:363 @ compile(): jit::compile: Encountered errors:
0: LLVMToCLSPV: clspv invocation failed with exit code -2
[AdaptiveCpp Error] from .../src/runtime/vk/vk_queue.cpp:885 @ submit_sscp_kernel_from_code_object(): vk_queue: Code object construction failed
data[n-1] = 0 (expected 64)
(followed by the "unhandled asynchronous errors" report and exit code 134)
```

OpenCL (Intel Arc, same GPU) and CUDA (NVIDIA, separate build) both print
`data[n-1] = 64 (expected 64)`.

HeCBench `scan-sycl` (`./main 16777216 100`) hits the same abort for several of
its kernels (e.g. `SrcTy = [32 x i32] ... CstVal = -8` and
`SrcTy = [64 x i32] ... CstVal = 16`).

**Expected behavior**

The kernel compiles for Vulkan and prints `data[n-1] = 64`, as with the other
backends.

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the Vulkan and OpenCL backends enabled; Vulkan SDK 1.4.341.1
* clspv built from source following `doc/install-vulkan.md`: upstream
  `a3d3e261` plus local commit `2ecc89b5` (see the note above)
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA: only for the comparison (separate build, LLVM 21.1.8, CUDA 13.4)
* OpenCL: Intel compute-runtime 26.05.037020 (comparison only)
* Vulkan drivers: Mesa 26.0.8 (ANV; llvmpipe with LLVM 21.1.8), NVIDIA 595.99.02
* Hardware/OS: Intel Core Ultra 7 155H with Intel Arc Graphics (Meteor Lake),
  NVIDIA RTX 500 Ada Generation Laptop GPU; Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Compiled with `acpp -O3 --acpp-targets=generic` (generic SSCP target)

**Optional additional diagnostic information**

`ACPP_S2_DUMP_IR_FINAL=1` shows the IR handed to clspv. In it, local memory is
the global `@__acpp_sscp_spirv_dynamic_local_mem.resized` of type
`[N x i32]`, and the `char` accesses are `getelementptr i8` instructions on it;
the failing ones have a constant byte offset that is not a multiple of 4.
`llvm-reduce` on one of the dumped scan modules gives:

```llvm
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-vulkan"

@__acpp_sscp_spirv_dynamic_local_mem.resized = external addrspace(3) global [32 x i32]

define spir_kernel void @k() {
entry:
  %0 = getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__acpp_sscp_spirv_dynamic_local_mem.resized, i64 -1), i64 0
  ret void
}
```

which aborts identically with the clspv options AdaptiveCpp uses
(`clspv -x=ir --physical-storage-buffers -spv-version=1.5 --arch=spir64
-long-vector -max-pushconstant-size=256 -max-ubo-size=65536`, after `llvm-as`).

**Additional context**

The abort is inside clspv; the second body below is a standalone clspv
reproducer. On a second machine (Radeon 780M; RADV and llvmpipe), HeCBench
`scan-sycl` also failed with `LLVMToCLSPV: clspv invocation failed with exit
code -2`; the clspv message was not recorded there.

---

## Body for the clspv issue

### `SimplifyPointerBitcastPass`: UNREACHABLE in `GetIdxsForTyFromOffset` for an `i8` GEP at an unaligned offset into an `i32` array

#### Description

`SimplifyPointerBitcastPass` aborts on a byte (`i8`) GEP into a global of type
`[N x i32]` whose constant offset is not a multiple of 4 when that GEP is used
as the base of another GEP:

```
Err: SrcTy = [32 x i32] - DstTy = i8 - Ty = [32 x i32] - CstVal = -8
Unexpected offset for type in GetIdxsForTyFromOffset
UNREACHABLE executed at lib/BitcastUtils.cpp:1443!
```

This IR is produced by AdaptiveCpp's SYCL-to-Vulkan path for `char` accesses
into local memory, which AdaptiveCpp declares as an `i32` array (found with
HeCBench `scan-sycl`).

#### Reproducer

`unaligned_i8_gep.ll`:

```llvm
target triple = "spir64-unknown-unknown"

@lm = internal addrspace(3) global [32 x i32] undef

define spir_kernel void @k() {
entry:
  %0 = getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @lm, i64 -1), i64 0
  ret void
}
```

```console
$ clspv-opt --passes=simplify-pointer-bitcast unaligned_i8_gep.ll -o out.ll
Err: SrcTy = [32 x i32] - DstTy = i8 - Ty = [32 x i32] - CstVal = -8
Unexpected offset for type in GetIdxsForTyFromOffset
UNREACHABLE executed at lib/BitcastUtils.cpp:1443!
Aborted (core dumped)
```

Varying the inner constant offset: 1, 2, 3, 5, −1, −2 and −3 bytes abort
(`CstVal` = 8 × offset); 4 and −4 bytes do not. The full `clspv` driver with
`-x=ir --physical-storage-buffers -spv-version=1.5 --arch=spir64 -long-vector`
aborts the same way on the original module.

#### Expected result

The pass handles (or leaves untouched) GEPs whose byte offset does not fall on
an element boundary, and clspv generates a SPIR-V module.

#### Backtrace

```
#5  llvm::llvm_unreachable_internal          lib/Support/ErrorHandling.cpp:244
#6  BitcastUtils::GetIdxsForTyFromOffset     lib/BitcastUtils.cpp:1443
#7  clspv::SimplifyPointerBitcastPass::runOnImplicitGEP   lib/SimplifyPointerBitcastPass.cpp:699
#8  clspv::SimplifyPointerBitcastPass::run    lib/SimplifyPointerBitcastPass.cpp:52
...
#15 clspv::Compile                            lib/Compiler.cpp:1459
```

clspv `a3d3e261` (plus an unrelated local change to `ReplacePointerBitcastPass`;
see the note at the top of this file).

---
