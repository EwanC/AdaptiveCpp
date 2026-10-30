# BUG-14: OpenCL `info::device::sub_group_sizes` lists sizes the device does not support

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) |
| Suggested title | `[OpenCL] info::device::sub_group_sizes returns every power of two up to the work-group size (1…512 on Intel GPU)` |
| Backend / devices | OpenCL (Intel GPU and CPU runtimes); found while comparing against the Vulkan backend |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-14, runs R14, R15 (M2); HeCBench `softmax-sycl` |
| Note | Not a Vulkan bug. The code documents this as a known heuristic (see Additional context). |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

On the OpenCL backend, `info::device::sub_group_sizes` returns all powers of
two from 1 up to the maximum work-group size: `1 2 4 … 512` for Intel Arc
Graphics and `1 2 4 … 4096` for the Intel OpenCL CPU device. The Intel OpenCL
runtimes report 8/16/32 (GPU) and 4/8/16/32/64 (CPU), and the kernels run here
used sizes from those lists (32 or 16 on the GPU depending on the kernel, 8 on
the CPU). Applications that size their work from the largest reported
sub-group size break: HeCBench `softmax-sycl` takes the maximum (512), computes
`256 / 512 == 0` and dies with SIGFPE (integer division by zero) in host code on
both OpenCL devices.

**To Reproduce**

`repro.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
  sycl::queue q;
  auto dev = q.get_device();
  printf("%s: sub_group_sizes =", dev.get_info<sycl::info::device::name>().c_str());
  for (auto s : dev.get_info<sycl::info::device::sub_group_sizes>()) printf(" %zu", s);
  printf("\n");
  int *out = sycl::malloc_device<int>(1, q);
  q.parallel_for(sycl::nd_range<1>{256, 256}, [=](sycl::nd_item<1> it) {
    if (it.get_global_linear_id() == 0) out[0] = it.get_sub_group().get_local_range()[0];
  }).wait();
  int h = 0;
  q.memcpy(&h, out, sizeof(int)).wait();
  printf("sub-group size used by the kernel: %d\n", h);
  sycl::free(out, q);
}
```

```sh
acpp -O3 --acpp-targets=generic repro.cpp -o repro
ACPP_VISIBILITY_MASK="omp;ocl:Arc" ./repro
ACPP_VISIBILITY_MASK="omp;ocl:Ultra" ./repro
clinfo | grep -E "Device Name|Sub-group sizes \(Intel\)"
```

Output:

```
Intel(R) Arc(TM) Graphics: sub_group_sizes = 1 2 4 8 16 32 64 128 256 512
sub-group size used by the kernel: 32
Intel(R) Core(TM) Ultra 7 155H: sub_group_sizes = 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
sub-group size used by the kernel: 8
```

(A kernel containing a `reduce_over_group` on the sub-group ran with sub-groups
of 16 on the same GPU.)

```
  Device Name                                     Intel(R) Arc(TM) Graphics
  Sub-group sizes (Intel)                         8, 16, 32
  Device Name                                     Intel(R) Core(TM) Ultra 7 155H
  Sub-group sizes (Intel)                         4, 8, 16, 32, 64
```

HeCBench (<https://github.com/zjin-lcf/HeCBench>, commit `8d66934f`)
`softmax-sycl` (`./main 100000 784 1 100`) on either OpenCL device exits with
SIGFPE (exit code 136); under gdb: `received signal SIGFPE, Arithmetic
exception` in `main()`. Its host code does

```cpp
auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
int warpSize = *std::max_element(sg_sizes.begin(), sg_sizes.end());
sycl::range<1> gws ((numSlice+BLOCK_SIZE/warpSize-1)/(BLOCK_SIZE/warpSize)*BLOCK_SIZE); // BLOCK_SIZE 256
```

**Expected behavior**

SYCL 2020 (rev. 12, §4.6.4.4): `info::device::sub_group_sizes` "Returns: A
`std::vector` of `std::size_t` containing the set of sub-group sizes supported
by the device." Here that would be 8, 16, 32 for the GPU and 4, 8, 16, 32, 64
for the CPU device, as reported by the OpenCL runtimes.

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the OpenCL and Vulkan backends enabled
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA / ROCm: not involved
* OpenCL: Intel compute-runtime (NEO) 26.05.037020 for Intel Arc Graphics
  (Meteor Lake); Intel OpenCL CPU runtime 2026.21.3.0.31; both expose
  `cl_intel_required_subgroup_size`
* Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Compiled with `acpp -O3 --acpp-targets=generic` (generic SSCP target)

**Optional additional diagnostic information**

`acpp-info` shows the same lists under `sub_group_sizes`.

**Additional context**

`ocl_hardware_context::get_property(device_uint_list_property::sub_group_sizes)`
(`src/runtime/ocl/ocl_hardware_manager.cpp`, line ~529 at `f4ff3681`) is a
documented heuristic: "there does not seem to be a direct query for this. So we
return all power of two values between 1 and the max workgroup size ... may
also return values that are not actually possible as subgroups." Both Intel
runtimes here expose `cl_intel_required_subgroup_size`, whose
`CL_DEVICE_SUB_GROUP_SIZES_INTEL` query is what `clinfo` prints above.

---
