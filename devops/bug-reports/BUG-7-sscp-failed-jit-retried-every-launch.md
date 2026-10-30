# BUG-7: A kernel whose JIT compilation fails is recompiled on every launch

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) |
| Suggested title | `[SSCP] Failed JIT compilations are not cached; every launch recompiles (application appears to hang)` |
| Backend / devices | Generic SSCP JIT path (kernel cache); observed on the Vulkan backend (all drivers) |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-7, runs R14, R15 (M2); HeCBench `scan-sycl` |
| Related | The compilation failure itself is [BUG-11](BUG-11-vk-clspv-unaligned-local-gep.md); this report is about what happens after it |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

When the JIT compilation of a kernel fails, the failure is not remembered.
Every later launch of the same kernel repeats the whole JIT pipeline,
including spawning the external compiler. On the Vulkan backend one failed
compilation (clspv aborting) costs ~1.7 s here, so an application that keeps
launching the kernel appears to hang: HeCBench `scan-sycl` performed 129–130
failed compilations in 180 s before being killed by a timeout. The errors are
only reported asynchronously, so `queue::wait()` returns normally and the
application keeps going; the process is terminated at exit with an
"unhandled asynchronous errors" report.

**To Reproduce**

`repro.cpp` (a work-efficient scan up-sweep; on the Vulkan backend clspv
currently fails on this kernel, see the related report):

```cpp
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>

int main() {
  sycl::queue q{sycl::property::queue::in_order()};
  const int n = 64;
  char *data = sycl::malloc_device<char>(n, q);
  for (int launch = 0; launch < 3; launch++) {
    q.memset(data, 1, n);
    auto t0 = std::chrono::steady_clock::now();
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
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    char last = 0;
    q.memcpy(&last, data + n - 1, 1).wait();
    printf("launch %d: %.0f ms, data[n-1] = %d (expected %d)\n", launch, ms, last, n);
  }
  sycl::free(data, q);
}
```

```sh
acpp -O3 --acpp-targets=generic repro.cpp -o repro
ACPP_VISIBILITY_MASK="omp;vk:NVIDIA" ./repro    # same on vk:MTL and vk:llvmpipe
ACPP_VISIBILITY_MASK="omp;ocl:Arc" ./repro      # for comparison
```

Vulkan output (abridged; the four error lines are printed once per launch):

```
Err: SrcTy = [16 x i32] - DstTy = i8 - Ty = [16 x i32] - CstVal = -8
Unexpected offset for type in GetIdxsForTyFromOffset
UNREACHABLE executed at .../clspv/lib/BitcastUtils.cpp:1443!
[AdaptiveCpp Error] from .../include/hipSYCL/glue/llvm-sscp/jit.hpp:363 @ compile(): jit::compile: Encountered errors:
0: LLVMToCLSPV: clspv invocation failed with exit code -2
[AdaptiveCpp Error] from .../src/runtime/vk/vk_queue.cpp:885 @ submit_sscp_kernel_from_code_object(): vk_queue: Code object construction failed
launch 0: 1756 ms, data[n-1] = 0 (expected 64)
  ... same errors again ...
launch 1: 1732 ms, data[n-1] = 0 (expected 64)
  ... same errors again ...
launch 2: 1724 ms, data[n-1] = 0 (expected 64)
============== AdaptiveCpp error report ==============
AdaptiveCpp has caught the following unhandled asynchronous errors:
   ...
The application will now be terminated.
terminate called without an active exception
(exit code 134)
```

clspv was started three times, once per launch, and each launch took ~1.7 s.
With the OpenCL backend on the same GPU (and with the CUDA backend), only the
first launch compiles:

```
launch 0: 245 ms, data[n-1] = 64 (expected 64)
launch 1: 0 ms, data[n-1] = 64 (expected 64)
launch 2: 0 ms, data[n-1] = 64 (expected 64)
```

**Expected behavior**

A failed compilation for a given kernel configuration should be reported once;
later launches of the same configuration should fail immediately (ideally with
a synchronous exception from the kernel submission) instead of rerunning the
JIT compiler.

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the Vulkan and OpenCL backends enabled; clspv built from source at
  `2ecc89b5` following `doc/install-vulkan.md`; Vulkan SDK 1.4.341.1
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA: only for the comparison (separate build, LLVM 21.1.8, CUDA 13.4)
* OpenCL: Intel compute-runtime 26.05.037020 (comparison only)
* Vulkan drivers: Mesa 26.0.8 (ANV; llvmpipe with LLVM 21.1.8), NVIDIA 595.99.02
* Hardware/OS: Intel Core Ultra 7 155H with Intel Arc Graphics (Meteor Lake),
  NVIDIA RTX 500 Ada Generation Laptop GPU; Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Compiled with `acpp -O3 --acpp-targets=generic` (generic SSCP target)

**Optional additional diagnostic information**

With `ACPP_DEBUG_LEVEL=3`, every launch logs
`kernel_cache: Cache MISS for id 10146517648093364908.9399881170374432198`
(the same id each time) and clspv prints its abort message again.

HeCBench `scan-sycl` (`./main 16777216 100`, Vulkan, 180 s timeout) logged 129–130
`Code object construction failed` errors per run on each of the three Vulkan
drivers and never finished.

**Additional context**

Relevant code at `f4ff3681`:
`kernel_cache::get_or_construct_jit_code_object()`
(`include/hipSYCL/runtime/kernel_cache.hpp`) returns `nullptr` when the
`jit_compile` callback fails, without recording anything for that
`id_of_code_object`; the next call misses the cache again. This function is
shared by the OpenCL, Level Zero, Metal and Vulkan queues.

---
