# BUG-6: Kernels needing unsupported SPIR-V capabilities are launched anyway

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) |
| Suggested title | `[VK] Kernels using unsupported SPIR-V capabilities (AtomicFloat64AddEXT) run and give wrong results / crash the driver` |
| Backend / devices | Vulkan; Intel ANV (wrong result), llvmpipe (process abort); NVIDIA works |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-6, runs R14, R15 (M2); HeCBench `atomicCost` FP64 sub-test |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

A kernel that uses `sycl::atomic_ref<double>::fetch_add` produces SPIR-V that
requires the `AtomicFloat64AddEXT` capability (6034). On Vulkan devices that
do not support `shaderBufferFloat64AtomicAdd`, the Vulkan backend prints
`vk_executable_object: unexpected SPIRV capability 6034` and launches the
kernel anyway:

* Intel Arc Graphics (Meteor Lake, Mesa ANV): the result is `-nan` instead of
  1024.
* llvmpipe: the driver's LLVM aborts the whole process
  (`LLVM ERROR: Cannot select: ... store<(store (s64)), trunc to f64> ...`).
* NVIDIA (supports the feature): correct result.

All three devices report `aspect::fp64 == false` and `aspect::atomic64 == false`
through the Vulkan backend.

**To Reproduce**

`repro.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
  sycl::queue q;
  auto dev = q.get_device();
  printf("%s: aspect::fp64=%d aspect::atomic64=%d\n",
         dev.get_info<sycl::info::device::name>().c_str(), dev.has(sycl::aspect::fp64),
         dev.has(sycl::aspect::atomic64));
  double *d = sycl::malloc_device<double>(1, q);
  q.memset(d, 0, sizeof(double)).wait();
  q.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1>) {
    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        a(*d);
    a.fetch_add(1.0);
  }).wait();
  double h = 0;
  q.memcpy(&h, d, sizeof(double)).wait();
  printf("result %g (expected 1024)\n", h);
  sycl::free(d, q);
}
```

```sh
acpp -O3 --acpp-targets=generic repro.cpp -o repro
ACPP_VISIBILITY_MASK="omp;vk:MTL" ./repro
ACPP_VISIBILITY_MASK="omp;vk:llvmpipe" ./repro
ACPP_VISIBILITY_MASK="omp;vk:NVIDIA" ./repro
```

Output:

```
Intel(R) Arc(tm) Graphics (MTL): aspect::fp64=0 aspect::atomic64=0
[AdaptiveCpp Error] from .../src/runtime/vk/vk_code_object.cpp:642 @ verify_spv_capabilities(): vk_executable_object: unexpected SPIRV capability 6034
result -nan (expected 1024)

llvmpipe (LLVM 21.1.8, 256 bits): aspect::fp64=0 aspect::atomic64=0
[AdaptiveCpp Error] from .../src/runtime/vk/vk_code_object.cpp:642 @ verify_spv_capabilities(): vk_executable_object: unexpected SPIRV capability 6034
LLVM ERROR: Cannot select: 0x776008216460: ch = store<(store (s64)), trunc to f64> 0x776008202d80, Constant:i64<0>, 0x776008216b60, undef:i64
In function: cs_variant
(process aborted, exit code 134)

NVIDIA RTX 500 Ada Generation Laptop GPU: aspect::fp64=0 aspect::atomic64=0
[AdaptiveCpp Error] from .../src/runtime/vk/vk_code_object.cpp:642 @ verify_spv_capabilities(): vk_executable_object: unexpected SPIRV capability 6034
result 1024 (expected 1024)
```

`vulkaninfo` for the same devices:

```
Intel(R) Arc(tm) Graphics (MTL):     shaderBufferFloat64AtomicAdd = false
NVIDIA RTX 500 Ada Generation ...:   shaderBufferFloat64AtomicAdd = true
llvmpipe (LLVM 21.1.8, 256 bits):    shaderBufferFloat64AtomicAdd = false
```

The CUDA backend on the same NVIDIA GPU reports `aspect::fp64=1 aspect::atomic64=1`
and also prints `result 1024`.

**Expected behavior**

The kernel should not be launched on a device that cannot execute it. SYCL 2020
(rev. 12, §5.7 "Optional kernel features") says that when an application submits
a kernel using an optional feature the device does not support, "the
implementation throws an exception with the `errc::kernel_not_supported` error
code from the kernel invocation command (e.g. `parallel_for()`)". At a minimum
the launch should fail with an error instead of producing a wrong result or
aborting the process inside the driver. On devices that do support the feature
(NVIDIA here) the kernel should keep working.

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the Vulkan and OpenCL backends enabled; clspv built from source at
  `2ecc89b5` following `doc/install-vulkan.md`; Vulkan SDK 1.4.341.1
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA: only for the comparison above (separate build, LLVM 21.1.8, CUDA 13.4,
  driver 595.99.02)
* OpenCL: not involved
* Vulkan drivers: Mesa 26.0.8 (ANV; llvmpipe with LLVM 21.1.8), NVIDIA 595.99.02
* Hardware/OS: Intel Core Ultra 7 155H with Intel Arc Graphics (Meteor Lake),
  NVIDIA RTX 500 Ada Generation Laptop GPU; Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Compiled with `acpp -O3 --acpp-targets=generic` (generic SSCP target)

**Optional additional diagnostic information**

The same failure shows up in HeCBench `atomicCost-sycl` (`./main 16 10`): the
FP64 sub-test fails on ANV and aborts on llvmpipe, and passes on NVIDIA.

**Additional context**

Relevant code at `f4ff3681`:

* `vk_executable_object::verify_spv_capabilities()`
  (`src/runtime/vk/vk_code_object.cpp`) checks a list of capabilities against
  device features; any capability not in the list reaches the `default:` branch,
  which only calls `print_error()`. Kernel construction then continues.
* `VK_EXT_shader_atomic_float` / `shaderBufferFloat64AtomicAdd` are not referenced
  anywhere in `src/runtime/vk` or `include/hipSYCL/runtime/vk`.
* `vk_hardware_context::has()` (`src/runtime/vk/vk_hardware_manager.cpp`)
  returns `false` for `device_support_aspect::fp64` and `atomic64`
  (comment: "Not supported by clspv"). On the same machine, FP64 kernels without
  atomics (e.g. HeCBench `su3-sycl`) compile and produce verified results on all
  three Vulkan drivers.

---
