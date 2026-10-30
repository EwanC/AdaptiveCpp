# BUG-13: OpenCL USM allocations fail after the runtime is re-initialised

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) |
| Suggested title | `[OpenCL] USM entry points cannot be resolved after the runtime is shut down and restarted in the same process` |
| Backend / devices | OpenCL (Intel GPU and CPU runtimes); found while comparing against the Vulkan backend |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-13, runs R14, R15 (M2); HeCBench `histogram`, `atomicCost`, `babelstream` |
| Note | Not a Vulkan bug. The Vulkan backend handles the same program. |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

If all SYCL objects are destroyed (so the AdaptiveCpp runtime shuts down) and
new ones are created later in the same process, the OpenCL backend can no
longer resolve the `cl_intel_unified_shared_memory` entry points:
`clGetExtensionFunctionAddressForPlatform` returns null for all of them,
`malloc_device` returns `nullptr` with `CL:-32` (`CL_INVALID_PLATFORM`), and the
process is terminated with an "unhandled asynchronous errors" report. HeCBench
benchmarks that create a queue per test configuration (`histogram-sycl`,
`atomicCost-sycl`, `babelstream-sycl`) crash on their second configuration.

**To Reproduce**

`repro.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
  for (int iter = 0; iter < 2; iter++) {
    sycl::queue q;
    int *p = sycl::malloc_device<int>(1024, q);
    printf("iteration %d: %s, malloc_device returned %p\n", iter,
           q.get_device().get_info<sycl::info::device::name>().c_str(), (void *)p);
    if (p) {
      q.memset(p, 0, 1024 * sizeof(int)).wait();
      sycl::free(p, q);
    }
  } // the queue is destroyed here; no SYCL objects remain
}
```

```sh
acpp -O3 --acpp-targets=generic repro.cpp -o repro
ACPP_VISIBILITY_MASK="omp;ocl:Arc" ./repro     # same with omp;ocl:Ultra
```

Output (paths shortened; the `initialize_func` error is printed once per entry
point, eleven times in total):

```
iteration 0: Intel(R) Arc(TM) Graphics, malloc_device returned 0xffffffffff9ff000
[AdaptiveCpp Error] from .../src/runtime/ocl/ocl_usm.cpp:313 @ initialize_func(): ocl_usm_extension: Platform advertises USM support, but extracting function address for clHostMemAllocINTEL failed.
[AdaptiveCpp Error] from .../src/runtime/ocl/ocl_usm.cpp:313 @ initialize_func(): ocl_usm_extension: Platform advertises USM support, but extracting function address for clDeviceMemAllocINTEL failed.
...
[AdaptiveCpp Error] from .../src/runtime/ocl/ocl_allocator.cpp:36 @ raw_allocate(): ocl_allocator: USM device allocation failed (error code = CL:-32)
iteration 1: Intel(R) Arc(TM) Graphics, malloc_device returned (nil)
============== AdaptiveCpp error report ==============
AdaptiveCpp has caught the following unhandled asynchronous errors:

   0. from .../src/runtime/ocl/ocl_allocator.cpp:36 @ raw_allocate(): ocl_allocator: USM device allocation failed (error code = CL:-32)
The application will now be terminated.
terminate called without an active exception
(exit code 134)
```

With `ACPP_VISIBILITY_MASK="omp;vk:MTL"` (same GPU, Vulkan backend) both
iterations succeed.

**Expected behavior**

The second iteration behaves like the first.

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the OpenCL and Vulkan backends enabled
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA / ROCm: not involved
* OpenCL: Intel compute-runtime (NEO) 26.05.037020 for Intel Arc Graphics
  (Meteor Lake); Intel OpenCL CPU runtime 2026.21.3.0.31; PoCL 7.2-pre is also
  installed (hidden by AdaptiveCpp: no SPIR-V support); ICD loader from Ubuntu
* Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Compiled with `acpp -O3 --acpp-targets=generic` (generic SSCP target)

**Optional additional diagnostic information**

Excerpt with `ACPP_DEBUG_LEVEL=3` (Intel GPU):

```
[AdaptiveCpp Info] backend_loader: Successfully opened plugin: ".../lib/hipSYCL/librt-backend-ocl.so"
[AdaptiveCpp Info] ocl_hardware_manager: Discovered OpenCL platform Portable Computing Language
[AdaptiveCpp Info] ocl_hardware_manager: Discovered OpenCL platform Intel(R) OpenCL Graphics
[AdaptiveCpp Info] ocl_hardware_manager: Discovered OpenCL platform Intel(R) OpenCL
[AdaptiveCpp Info] runtime: ******* rt launch initiated ********
iteration 0: Intel(R) Arc(TM) Graphics, malloc_device returned 0xffffffffff9ff000
[AdaptiveCpp Info] runtime: ******* rt shutdown ********
[AdaptiveCpp Info] backend_loader: Successfully opened plugin: ".../lib/hipSYCL/librt-backend-ocl.so"
[AdaptiveCpp Info] ocl_hardware_manager: Discovered OpenCL platform Portable Computing Language
[AdaptiveCpp Info] ocl_hardware_manager: Discovered OpenCL platform Intel(R) OpenCL Graphics
[AdaptiveCpp Error] ... ocl_usm_extension: Platform advertises USM support, but extracting function address for clHostMemAllocINTEL failed.
...
```

**Additional context**

The failing call is `clGetExtensionFunctionAddressForPlatform(id, name)` in
`ocl_usm_extension::initialize_func()` (`src/runtime/ocl/ocl_usm.cpp`, line ~310
at `f4ff3681`); it succeeds during the first runtime initialisation of the
process and returns null for every USM entry point during the second.

---
