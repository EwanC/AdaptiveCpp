# BUG-10: Intel ANV runs kernels with a different sub-group size than reported

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) |
| Suggested title | `[VK] Sub-group operations give wrong results on Intel (ANV): kernels run SIMD16 while sub-group size 32 is reported` |
| Backend / devices | Vulkan; Intel Arc Graphics (Meteor Lake) with Mesa ANV. NVIDIA and llvmpipe are not affected |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-5 (softmax, layernorm rows), BUG-10; runs R14, R15 (M2) |
| Fix | Branch `vk/per-kernel-subgroup-size` (`055fe70d`, "[VK] Fix the sub-group size of kernels") fixes the reproducer and both benchmarks. File this issue only if it is useful to link from that PR. |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

On Intel Arc Graphics (Meteor Lake) with Mesa ANV, the Vulkan backend reports
a sub-group size of 32 (`info::device::sub_group_sizes == {32}`), but ANV
compiles the kernels below as SIMD16. Sub-group queries and algorithms then give
inconsistent results: in the reproducer, 16 of 256 work-items see
`sub_group::get_local_range()` as a wrapped-around value (-224 when printed as
`int`). HeCBench `softmax-sycl` and `layernorm-sycl` produce wrong results on
this device; both pass on NVIDIA and llvmpipe through the Vulkan backend.

**To Reproduce**

`repro.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>
#include <map>
#include <vector>

int main() {
  sycl::queue q;
  auto dev = q.get_device();
  auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  printf("%s: sub_group_sizes =", dev.get_info<sycl::info::device::name>().c_str());
  for (auto s : sizes) printf(" %zu", s);
  printf("\n");

  const size_t wg = 256;
  int *out = sycl::malloc_device<int>(2 * wg, q);
  q.parallel_for(sycl::nd_range<1>{wg, wg}, [=](sycl::nd_item<1> it) {
    auto sg = it.get_sub_group();
    // Number of work-items that actually take part in a sub-group reduction
    int participants = sycl::reduce_over_group(sg, 1, sycl::plus<int>());
    size_t i = it.get_global_linear_id();
    out[2 * i] = participants;
    out[2 * i + 1] = sg.get_local_range()[0];
  }).wait();
  std::vector<int> h(2 * wg);
  q.memcpy(h.data(), out, h.size() * sizeof(int)).wait();
  std::map<std::pair<int, int>, int> seen;
  bool ok = true;
  for (size_t i = 0; i < wg; i++) {
    seen[{h[2 * i], h[2 * i + 1]}]++;
    ok &= h[2 * i] == h[2 * i + 1];
  }
  for (auto [k, count] : seen)
    printf("  %3d work-items: reduce_over_group(sg, 1) = %d, sg.get_local_range() = %d\n",
           count, k.first, k.second);
  printf("%s\n", ok ? "PASS" : "FAIL: reduction result differs from the sub-group size");
  sycl::free(out, q);
}
```

```sh
acpp -O3 --acpp-targets=generic repro.cpp -o repro
ACPP_VISIBILITY_MASK="omp;vk:MTL" ./repro
```

Output on Intel Arc Graphics (MTL), ANV:

```
Intel(R) Arc(tm) Graphics (MTL): sub_group_sizes = 32
   16 work-items: reduce_over_group(sg, 1) = 32, sg.get_local_range() = -224
  240 work-items: reduce_over_group(sg, 1) = 32, sg.get_local_range() = 32
FAIL: reduction result differs from the sub-group size
```

Output on NVIDIA (proprietary driver) and llvmpipe through the Vulkan backend:

```
NVIDIA RTX 500 Ada Generation Laptop GPU: sub_group_sizes = 32
  256 work-items: reduce_over_group(sg, 1) = 32, sg.get_local_range() = 32
PASS
llvmpipe (LLVM 21.1.8, 256 bits): sub_group_sizes = 8
  256 work-items: reduce_over_group(sg, 1) = 8, sg.get_local_range() = 8
PASS
```

With `MESA_SHADER_CACHE_DISABLE=true INTEL_DEBUG=cs`, ANV reports the
reproducer's kernel as `SIMD16 shader: 100 instructions ...`.

HeCBench (<https://github.com/zjin-lcf/HeCBench>, commit `8d66934f`) on the
same device:

* `softmax-sycl` (`./main 100000 784 1 100`): `@index 19 host: 0.001318 device: 0.000000`, `FAIL`
* `layernorm-sycl` (`./main 8 1024 12288 200`): `Mismatch of out at 6: CPU_ref: 0.15959 vs GPU: 0.72517` (and more), exit code 1

**Expected behavior**

Kernels run with the sub-group size that the device reports and that the
generated code assumes, so that the reproducer prints `PASS` (256 work-items
with 32/32) and both benchmarks verify, as on the other Vulkan devices.

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the Vulkan and OpenCL backends enabled; clspv built from source
  (upstream `a3d3e261` plus an unrelated local commit) following
  `doc/install-vulkan.md`; Vulkan SDK 1.4.341.1
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA / ROCm / OpenCL: not involved
* Vulkan drivers: Mesa 26.0.8 (ANV; llvmpipe with LLVM 21.1.8), NVIDIA 595.99.02
* Hardware/OS: Intel Core Ultra 7 155H with Intel Arc Graphics (Meteor Lake),
  NVIDIA RTX 500 Ada Generation Laptop GPU; Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Compiled with `acpp -O3 --acpp-targets=generic` (generic SSCP target)

**Optional additional diagnostic information**

`vulkaninfo` for the Intel device: `subgroupSize = 32`, `minSubgroupSize = 8`,
`maxSubgroupSize = 32`, `subgroupSizeControl = true`, and compute is listed in
`requiredSubgroupSizeStages`.

**Additional context**

With the Vulkan backend from branch `vk/per-kernel-subgroup-size` (`055fe70d`),
which requires the selected sub-group size through
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`, the same reproducer
prints `256 work-items: reduce_over_group(sg, 1) = 32, sg.get_local_range() = 32`
and `PASS`, ANV reports the kernel as `SIMD32 shader`, and `softmax-sycl` and
`layernorm-sycl` both pass.

---
