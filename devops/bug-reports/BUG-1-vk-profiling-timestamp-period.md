# BUG-1: Event profiling durations are wrong when `timestampPeriod` ≠ 1

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) |
| Suggested title | `[VK] Event profiling durations are not scaled by VkPhysicalDeviceLimits::timestampPeriod` |
| Backend / devices | Vulkan; Intel ANV (Meteor Lake), AMD RADV (Radeon 780M) |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-1, runs R2 (M1), R13 (M2) |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

On the Vulkan backend, durations computed from
`event::get_profiling_info<command_start/command_end>()` are far too small on
devices whose `VkPhysicalDeviceLimits::timestampPeriod` is not 1 ns:

* Intel Arc Graphics (Meteor Lake, Mesa ANV, `timestampPeriod` = 52.083): a
  kernel that takes ~4.3 ms of wall-clock time is reported as 127 µs.
* Radeon 780M (Mesa RADV, `timestampPeriod` = 10): a ~6.8 ms kernel was
  reported as 684 µs.

On devices with `timestampPeriod` = 1 (NVIDIA proprietary driver, llvmpipe)
the reported durations are plausible.

**To Reproduce**

`repro.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>

int main() {
  sycl::queue q{sycl::default_selector_v,
                {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};
  const size_t n = 16 * 1024 * 1024;
  float *b = sycl::malloc_device<float>(n, q);
  auto k = [&] {
    return q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
      float x = i[0];
      for (int r = 0; r < 64; r++) x = x * 1.0001f + 0.5f;
      b[i] = x;
    });
  };
  k().wait(); // JIT compilation
  auto t0 = std::chrono::steady_clock::now();
  sycl::event e = k();
  e.wait();
  double wall = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  double prof = (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                 e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-3;
  printf("%s: wall clock %.0f us, event profiling %.0f us\n",
         q.get_device().get_info<sycl::info::device::name>().c_str(), wall, prof);
  sycl::free(b, q);
}
```

```sh
acpp -O3 --acpp-targets=generic repro.cpp -o repro
ACPP_VISIBILITY_MASK="omp;vk:MTL" ./repro
ACPP_VISIBILITY_MASK="omp;vk:NVIDIA" ./repro
ACPP_VISIBILITY_MASK="omp;vk:llvmpipe" ./repro
```

Output:

```
Intel(R) Arc(tm) Graphics (MTL): wall clock 4344 us, event profiling 127 us
NVIDIA RTX 500 Ada Generation Laptop GPU: wall clock 1711 us, event profiling 1392 us
llvmpipe (LLVM 21.1.8, 256 bits): wall clock 29267 us, event profiling 28287 us
```

`vulkaninfo` for the three devices:

```
deviceName      = Intel(R) Arc(tm) Graphics (MTL)
timestampPeriod = 52.0833
deviceName      = NVIDIA RTX 500 Ada Generation Laptop GPU
timestampPeriod = 1
deviceName      = llvmpipe (LLVM 21.1.8, 256 bits)
timestampPeriod = 1
```

A longer kernel (`x*1.0001f+0.5f`, 64 MB of data, 10 measured launches) gives
3285–3425 µs wall clock against 74.7–78.5 µs from event profiling on the same
Intel device.

**Expected behavior**

The profiled duration should be close to, and not larger than, the
wall-clock time of `submit` + `wait`, as on the other devices. For comparison,
the CUDA backend on the same NVIDIA GPU (separate AdaptiveCpp build of the same
commit) reports `wall clock 792 us, event profiling 768 us`.

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the Vulkan and OpenCL backends enabled; clspv built from source at
  `2ecc89b5` following `doc/install-vulkan.md`; Vulkan SDK 1.4.341.1
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA: only for the comparison above (LLVM 21.1.8, CUDA 13.4, driver 595.99.02)
* OpenCL: not involved
* Vulkan drivers: Mesa 26.0.8 (ANV; llvmpipe with LLVM 21.1.8), NVIDIA 595.99.02
* Hardware/OS: Intel Core Ultra 7 155H with Intel Arc Graphics (Meteor Lake),
  NVIDIA RTX 500 Ada Generation Laptop GPU; Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Compiled with `acpp -O3 --acpp-targets=generic` (generic SSCP target)

The RADV observation is from a second machine: AMD Ryzen 7 PRO 8840HS with
Radeon 780M, Mesa 25.2.8, Ubuntu 24.04, Linux 6.17, same AdaptiveCpp commit.

**Optional additional diagnostic information**

`acpp-info -l` lists the Vulkan devices as `Intel(R) Arc(tm) Graphics (MTL)`,
`NVIDIA RTX 500 Ada Generation Laptop GPU` and
`llvmpipe (LLVM 21.1.8, 256 bits)`.

**Additional context**

Relevant code at `f4ff3681`:

* `include/hipSYCL/runtime/vk/vk_profiling.hpp`,
  `vk_async_timestamp::get_time_point()` combines the `vkGetQueryPoolResults`
  value (`device_time`) with a calibrated `(clock_monotonic, device)`
  reference pair taken when the instrumentation object is constructed, and
  returns `host_ref - dev_ref + device_time` (or the equivalent `else`
  branch) as nanoseconds.
* `timestampPeriod` is only read in `src/runtime/vk/vk_hardware_manager.cpp`
  (to decide whether `execution_timestamps` is supported); it is not used
  when converting query results.

---
