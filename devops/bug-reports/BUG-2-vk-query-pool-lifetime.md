# BUG-2: Query pool destroyed while still in use when profiled events are dropped

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) |
| Suggested title | `[VK] Profiling query pools are destroyed while the command buffer is still executing` |
| Backend / devices | Vulkan; all drivers (crash on llvmpipe, device lost on RADV, validation error everywhere) |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-2, runs R2 (M1), R13 (M2) |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

With a queue constructed with `property::queue::enable_profiling`, submitting
kernels without keeping the returned events makes the Vulkan backend destroy
`VkQueryPool`s that are still referenced by executing command buffers. The
validation layer reports `VUID-vkDestroyQueryPool-queryPool-00793` on every
driver tried. llvmpipe then segfaults, aborts or hangs; on a Radeon 780M
(RADV) the same pattern ended in `The CS has been rejected` followed by
`vk::DeviceLostError` (timing dependent). Intel ANV and NVIDIA complete the
program.

**To Reproduce**

`repro.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
  sycl::queue q{sycl::default_selector_v,
                {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};
  printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  int *d = sycl::malloc_device<int>(1 << 20, q);
  for (int i = 0; i < 2000; i++)
    q.parallel_for(sycl::range<1>{1 << 20}, [=](sycl::id<1> i) { d[i] = d[i] * 3 + 1; });
  q.wait();
  sycl::free(d, q);
  printf("done\n");
}
```

```sh
acpp -O3 --acpp-targets=generic repro.cpp -o repro
# with the Khronos validation layer from the Vulkan SDK
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ACPP_VISIBILITY_MASK="omp;vk:NVIDIA" ./repro
# without layers
ACPP_VISIBILITY_MASK="omp;vk:llvmpipe" ./repro
```

Validation output (same message on ANV, NVIDIA and llvmpipe, repeated):

```
Validation Error: [ VUID-vkDestroyQueryPool-queryPool-00793 ] | MessageID = 0x5ef3bf2b
vkDestroyQueryPool(): can't be called on VkQueryPool 0x50000000005 that is currently in use by VkCommandBuffer 0x588389999570.
The Vulkan spec states: All submitted commands that refer to queryPool must have completed execution (https://vulkan.lunarg.com/doc/view/1.4.341.1/linux/antora/spec/latest/chapters/queries.html#VUID-vkDestroyQueryPool-queryPool-00793)
```

Outcome per driver:

| Device | With validation layer | Without layers (3 runs) |
|---|---|---|
| Intel Arc Graphics (MTL), ANV | VUID-00793, completes | completes |
| NVIDIA RTX 500 Ada, proprietary driver | VUID-00793, completes | completes |
| llvmpipe | VUID-00793, then SIGSEGV | SIGABRT once, hang (300 s timeout) twice |
| Radeon 780M, RADV (second machine) | — | `The CS has been rejected`, `vk::DeviceLostError` (timing dependent) |

On the first machine (RADV, below) the llvmpipe segfault was in the
`libvulkan_lvp.so` queue thread.

**Expected behavior**

No validation errors and no crash: query pools should stay alive until the
command buffers that use them have completed, whether or not the application
keeps the events.

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the Vulkan and OpenCL backends enabled; clspv built from source at
  `2ecc89b5` following `doc/install-vulkan.md`; Vulkan SDK 1.4.341.1
  (loader and `VK_LAYER_KHRONOS_validation`)
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA / ROCm / OpenCL: not involved
* Vulkan drivers: Mesa 26.0.8 (ANV; llvmpipe with LLVM 21.1.8), NVIDIA 595.99.02
* Hardware/OS: Intel Core Ultra 7 155H with Intel Arc Graphics (Meteor Lake),
  NVIDIA RTX 500 Ada Generation Laptop GPU; Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Compiled with `acpp -O3 --acpp-targets=generic` (generic SSCP target)

The RADV result is from a second machine: AMD Ryzen 7 PRO 8840HS with
Radeon 780M, Mesa 25.2.8, Ubuntu 24.04, Linux 6.17, same AdaptiveCpp commit.

**Optional additional diagnostic information**

The validation error does not appear when every event returned by
`parallel_for` is waited on before it is destroyed.

**Additional context**

Relevant code at `f4ff3681`: each profiled operation gets
`vk_execution_start_timestamp` / `vk_execution_finish_timestamp` objects
(`include/hipSYCL/runtime/vk/vk_profiling.hpp`), created in
`vk_queue::profile_if_enabled` (`src/runtime/vk/vk_queue.cpp`) and held through
`std::shared_ptr` in the operation's instrumentation set. Each of these objects
owns a single-query `vk::raii::QueryPool` (`_query_pool`) that
`begin_command_buffer` / `end_command_buffer` record `vkCmdWriteTimestamp` into.

---
