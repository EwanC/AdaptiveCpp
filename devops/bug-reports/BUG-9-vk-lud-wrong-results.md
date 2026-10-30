# BUG-9: HeCBench lud gives wrong results on every Vulkan driver

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) |
| Suggested title | `[VK] HeCBench lud-sycl fails verification on the Vulkan backend (ANV, NVIDIA, llvmpipe, RADV)` |
| Backend / devices | Vulkan; all four drivers tried |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-5 (lud rows), BUG-9; runs R7 (M1), R15 (M2) |
| Note | Not reduced to a minimal kernel yet. The OpenMP backend also fails this benchmark, with a different pattern; that is not covered here. |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

HeCBench `lud-sycl` with verification enabled (`-v`) reports ~32,000
mismatching matrix elements on the Vulkan backend with every driver tried
(Intel ANV, NVIDIA, llvmpipe, AMD RADV). The same binary passes verification
with the OpenCL backend on the same Intel GPU, and HIP passed on the second
machine.

**To Reproduce**

HeCBench (<https://github.com/zjin-lcf/HeCBench>) at commit `8d66934f`:

```sh
cd HeCBench/src/lud-sycl
make CC=acpp VENDOR=AdaptiveCpp GPU=yes EXTRA_CFLAGS="--acpp-targets=generic"
mv main lud-gpu && make clean
make CC=acpp VENDOR=AdaptiveCpp GPU=no  EXTRA_CFLAGS="--acpp-targets=generic"
mv main lud-cpu      # cpu_selector variant, needed for llvmpipe

ACPP_VISIBILITY_MASK="omp;ocl:Arc"      ./lud-gpu -s 1024 -v | grep -c mismatch
ACPP_VISIBILITY_MASK="omp;vk:MTL"       ./lud-gpu -s 1024 -v | grep -c mismatch
ACPP_VISIBILITY_MASK="omp;vk:NVIDIA"    ./lud-gpu -s 1024 -v | grep -c mismatch
ACPP_VISIBILITY_MASK="omp;vk:llvmpipe"  ./lud-cpu -s 1024 -v | grep -c mismatch
```

Results:

| Backend / device | mismatches | first mismatches printed |
|---|---|---|
| OpenCL, Intel Arc Graphics (MTL) | 0 | — |
| **Vulkan, Intel Arc Graphics (MTL), ANV** | **32286** | `mismatch at (1, 16): (o)9.851120 (n)105.597740`, `(1, 17)`, `(1, 18)`, … |
| **Vulkan, NVIDIA RTX 500 Ada** | **32314** | `mismatch at (1, 16): (o)9.851120 (n)107.046509`, `(1, 17)`, … |
| **Vulkan, llvmpipe** | **31368** | `mismatch at (1, 8): (o)9.930244 (n)19.840649`, `(1, 9)`, … |

The program exits normally in all cases and prints no AdaptiveCpp errors.

**Expected behavior**

No mismatches, as with the OpenCL backend.

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the Vulkan and OpenCL backends enabled; clspv built from source
  (upstream `a3d3e261` plus an unrelated local commit) following
  `doc/install-vulkan.md`; Vulkan SDK 1.4.341.1
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA / ROCm: not involved
* OpenCL: Intel compute-runtime 26.05.037020 (comparison)
* Vulkan drivers: Mesa 26.0.8 (ANV; llvmpipe with LLVM 21.1.8), NVIDIA 595.99.02
* Hardware/OS: Intel Core Ultra 7 155H with Intel Arc Graphics (Meteor Lake),
  NVIDIA RTX 500 Ada Generation Laptop GPU; Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Second machine: AMD Ryzen 7 PRO 8840HS, Radeon 780M, Mesa 25.2.8,
  Ubuntu 24.04: RADV and llvmpipe also report mismatches, HIP passes
* Benchmark compiled with `acpp --acpp-targets=generic -O3` through its Makefile

**Optional additional diagnostic information**

On the GPUs the first mismatch is at column 16 of row 1 (the benchmark's
`BLOCK_SIZE` is 16); on llvmpipe, whose sub-group size is 8, it is at column 8.

**Additional context**

The benchmark runs three kernels per block step: `diagonal` (work-group of 16),
`perimeter` (work-group of 32) and `internal` (16×16 work-group); `perimeter`
uses three `local_accessor<float, 1>` of 256 elements each
(`src/lud-sycl/lud.cpp`, lines 112–170 at the commit above). It has not been
reduced further.

---
