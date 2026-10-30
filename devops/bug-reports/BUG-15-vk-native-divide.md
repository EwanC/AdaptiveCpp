# BUG-15: `sycl::native::divide` makes the kernel fail to load on Vulkan

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) |
| Suggested title | `[VK] Kernels calling sycl::native::divide fail with "failed to parse spirv"` |
| Backend / devices | Vulkan; Intel ANV, NVIDIA and llvmpipe (all drivers tried) |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-15, run R18 (M2) |
| Note | Found by code inspection while checking user-facing performance advice, then reproduced. The IR experiment under *Optional additional diagnostic information* shows that a call with both operands gives a valid module, so this may be quicker to fix directly in a PR. No existing issue mentions `native_divide` (searched 2026-10-30); the code is unchanged on `develop` as of that date. |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

On the Vulkan backend, a kernel that calls `sycl::native::divide` cannot be
loaded: the runtime reports `failed to parse spirv` and
`Unknown kernel name ...`, the kernel does not run (its other outputs are not
written either), and the program is terminated with an "unhandled asynchronous
errors" report. This happens with all three Vulkan drivers tried. The same
program gives the expected results on the OpenCL (GPU and CPU), OpenMP and CUDA
backends.

The other `sycl::native` functions tested on the same devices (`recip`,
`rsqrt`, `sqrt`, `powr`, `exp`, `log`, `cos`) give the expected results.

**To Reproduce**

`repro.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
  sycl::queue q;
  const int n = 4;
  float hx[n] = {1.0f, 3.0f, 10.0f, -6.0f};
  float hy[n] = {2.0f, 4.0f, 5.0f, 3.0f};
  float *x = sycl::malloc_device<float>(n, q);
  float *y = sycl::malloc_device<float>(n, q);
  float *out = sycl::malloc_device<float>(2 * n, q);
  q.memcpy(x, hx, sizeof(hx));
  q.memcpy(y, hy, sizeof(hy)).wait();
  q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
    out[2 * i] = x[i] / y[i];
    out[2 * i + 1] = sycl::native::divide(x[i], y[i]);
  }).wait();
  float h[2 * n];
  q.memcpy(h, out, sizeof(h)).wait();
  printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  for (int i = 0; i < n; i++)
    printf("  %g / %g: operator/ = %g, sycl::native::divide = %g\n", hx[i], hy[i], h[2 * i],
           h[2 * i + 1]);
  sycl::free(x, q);
  sycl::free(y, q);
  sycl::free(out, q);
}
```

```sh
acpp -O3 --acpp-targets=generic repro.cpp -o repro
ACPP_VISIBILITY_MASK="omp;vk:MTL" ./repro       # same with vk:NVIDIA and vk:llvmpipe
ACPP_VISIBILITY_MASK="omp;ocl:Arc" ./repro      # for comparison
```

Vulkan output (Intel Arc Graphics, ANV; NVIDIA and llvmpipe print the same
errors):

```
[AdaptiveCpp Error] from .../src/runtime/vk/vk_code_object.cpp:679 @ vk_executable_object(): failed to parse spirv
[AdaptiveCpp Error] from .../src/runtime/vk/vk_code_object.cpp:689 @ vk_executable_object(): failed to parse spirv
[AdaptiveCpp Error] from .../src/runtime/vk/vk_code_object.cpp:827 @ get_kernel(): vk_executable_object: Unknown kernel name _Z18__acpp_sscp_kernelIN7hipsycl4glue15__sscp_dispatch18basic_parallel_forIZ4mainE3$_0Li1EEEEvRKT_
Intel(R) Arc(tm) Graphics (MTL)
  1 / 2: operator/ = 0, sycl::native::divide = 0
  3 / 4: operator/ = 0, sycl::native::divide = 0
  10 / 5: operator/ = 0, sycl::native::divide = 0
  -6 / 3: operator/ = 0, sycl::native::divide = 0
============== AdaptiveCpp error report ==============
AdaptiveCpp has caught the following unhandled asynchronous errors:

   0. from .../src/runtime/vk/vk_code_object.cpp:827 @ get_kernel(): vk_executable_object: Unknown kernel name _Z18__acpp_sscp_kernelIN7hipsycl4glue15__sscp_dispatch18basic_parallel_forIZ4mainE3$_0Li1EEEEvRKT_
The application will now be terminated.
terminate called without an active exception
(exit code 134)
```

OpenCL (Intel Arc Graphics, same GPU), and likewise OpenCL on the Intel CPU,
OpenMP and CUDA:

```
Intel(R) Arc(TM) Graphics
  1 / 2: operator/ = 0.5, sycl::native::divide = 0.5
  3 / 4: operator/ = 0.75, sycl::native::divide = 0.75
  10 / 5: operator/ = 2, sycl::native::divide = 2
  -6 / 3: operator/ = -2, sycl::native::divide = -2
```

**Expected behavior**

The kernel runs and both columns print `0.5`, `0.75`, `2`, `-2`, as on the
other backends.

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the Vulkan and OpenCL backends enabled; clspv built from source
  (upstream `a3d3e261` plus an unrelated local commit) following
  `doc/install-vulkan.md`; Vulkan SDK 1.4.341.1
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA: only for the comparison (separate AdaptiveCpp build of the same commit,
  LLVM 21.1.8, CUDA 13.4, driver 595.99.02)
* OpenCL: Intel compute-runtime 26.05.037020 (GPU), Intel OpenCL CPU runtime
  2026.21.3.0.31 (comparison only)
* Vulkan drivers: Mesa 26.0.8 (ANV; llvmpipe with LLVM 21.1.8), NVIDIA 595.99.02
* Hardware/OS: Intel Core Ultra 7 155H with Intel Arc Graphics (Meteor Lake),
  NVIDIA RTX 500 Ada Generation Laptop GPU; Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Compiled with `acpp -O3 --acpp-targets=generic` (generic SSCP target)

**Optional additional diagnostic information**

With `ACPP_S2_DUMP_IR_FINAL=1` (NVIDIA device), the IR handed to clspv calls
`native_divide` with one operand:

```llvm
declare dso_local spir_func noundef float @_Z13native_dividef(float noundef) local_unnamed_addr #1
...
  %15 = fdiv contract float %12, %14
  ...
  %17 = load float, ptr addrspace(1) %11, align 4, !tbaa !17, !alias.scope !12, !noalias !21
  %18 = call spir_func noundef float @_Z13native_dividef(float noundef %17) #2, !noalias !16
```

Running clspv by hand on this module (after `llvm-as`, with the options
AdaptiveCpp uses: `-x=ir --physical-storage-buffers -spv-version=1.5
--arch=spir64 -long-vector -max-pushconstant-size=256 -max-ubo-size=65536`)
exits with status 0, but the output does not validate:

```
$ spirv-val --target-env vulkan1.2 out.spv
error: line 108: Id is 0
```

Instruction 108 of that module is an `OpFDiv` whose second operand id is 0.
After changing the call in the dumped IR to the two-operand form (and the
declaration to match),

```llvm
  %18 = call spir_func noundef float @_Z13native_divideff(float noundef %17, float noundef %14) #2, !noalias !16
```

clspv produces a module that passes `spirv-val`, and that instruction's second
operand is a valid id.

**Additional context**

At `f4ff3681` (and on `develop` as of 2026-10-30),
`src/libkernel/sscp/clspv/native.cpp` lines 20–23 read:

```cpp
float native_divide(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_divide_f32(float x, float y) {
  return native_divide(x);
}
```

`native_powr`, the other two-operand function in the same file, is declared
and called with two operands.

---
