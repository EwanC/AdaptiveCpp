# BUG-12: `enable_profiling` on an OpenCL device without profiling support does not throw

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) |
| Suggested title | `[OpenCL] Queue with enable_profiling is accepted on devices without aspect::queue_profiling; get_profiling_info aborts later` |
| Backend / devices | OpenCL (Intel GPU and CPU runtimes); found while comparing against the Vulkan backend |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-12, run R13 (M2) |
| Note | Not a Vulkan bug. Minor conformance issue: the device correctly reports `aspect::queue_profiling == false`. |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

On the OpenCL backend, constructing a queue with
`property::queue::enable_profiling` succeeds even though the device reports
`aspect::queue_profiling == false`. The failure only surfaces later, when
`event::get_profiling_info` throws
`Operation not profiled: No execution start timestamp available` (an uncaught
exception in most programs).

**To Reproduce**

`repro.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
  sycl::queue q{sycl::property::queue::enable_profiling()};
  printf("%s: aspect::queue_profiling = %d\n",
         q.get_device().get_info<sycl::info::device::name>().c_str(),
         (int)q.get_device().has(sycl::aspect::queue_profiling));
  sycl::event e = q.single_task([] {});
  e.wait();
  auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  printf("command_start = %llu\n", (unsigned long long)start);
}
```

```sh
acpp -O3 --acpp-targets=generic repro.cpp -o repro
ACPP_VISIBILITY_MASK="omp;ocl:Arc" ./repro
ACPP_VISIBILITY_MASK="omp;ocl:Ultra" ./repro
```

Output:

```
Intel(R) Arc(TM) Graphics: aspect::queue_profiling = 0
terminate called after throwing an instance of 'hipsycl::sycl::exception'
  what():  Operation not profiled: No execution start timestamp available
(exit code 134)

Intel(R) Core(TM) Ultra 7 155H: aspect::queue_profiling = 0
terminate called after throwing an instance of 'hipsycl::sycl::exception'
  what():  Operation not profiled: No execution start timestamp available
(exit code 134)
```

(Each abort is followed by a `PLEASE submit a bug report to
https://software.intel.com/...` line printed by the Intel OpenCL runtime's crash
handler.)

**Expected behavior**

SYCL 2020 (rev. 12, §4.6.5.5, `property::queue::enable_profiling`): "If the
queue's associated device does not have `aspect::queue_profiling`, passing this
property to the queue's constructor causes the constructor to throw a
synchronous exception with the `errc::feature_not_supported` error code."

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the OpenCL and Vulkan backends enabled
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA / ROCm: not involved
* OpenCL: Intel compute-runtime (NEO) 26.05.037020 for Intel Arc Graphics
  (Meteor Lake); Intel OpenCL CPU runtime 2026.21.3.0.31 for the Core Ultra 7 155H
* Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Compiled with `acpp -O3 --acpp-targets=generic` (generic SSCP target)

**Optional additional diagnostic information**

The same program on a Vulkan device that supports timestamps prints
`aspect::queue_profiling = 1` and a `command_start` value.

**Additional context**

At `f4ff3681`, `src/runtime/ocl/` contains no instrumentation/profiling code
(no references to `instrumentation`, `profil` or `timestamp` in
`ocl_queue.cpp`), so the aspect value is consistent with the implementation; it
is only the constructor check that is missing.

---
