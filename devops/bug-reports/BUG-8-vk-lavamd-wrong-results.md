# BUG-8: HeCBench lavaMD gives wrong results on every Vulkan driver

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) |
| Suggested title | `[VK] HeCBench lavaMD-sycl produces wrong results on the Vulkan backend (ANV, NVIDIA, llvmpipe, RADV)` |
| Backend / devices | Vulkan; all four drivers tried |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-5 (lavaMD rows), BUG-8; runs R7 (M1), R15 (M2) |
| Note | Not reduced to a minimal kernel yet; the reproducer is the public benchmark. |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

HeCBench `lavaMD-sycl` produces wrong forces on the Vulkan backend with every
driver tried (Intel ANV, NVIDIA, llvmpipe, AMD RADV): the maximum relative
difference to the CUDA result is 9.9–45, whereas the OpenCL backend (on the
same Intel GPU and on the Intel CPU), the OpenMP backend and HIP agree with
CUDA to ~1e-5.

**To Reproduce**

HeCBench (<https://github.com/zjin-lcf/HeCBench>) at commit `8d66934f`, built
with the benchmark's `-DOUTPUT` option so that the results are written to
`result.txt`:

```sh
cd HeCBench/src/lavaMD-sycl
make CC=acpp VENDOR=AdaptiveCpp GPU=yes EXTRA_CFLAGS="--acpp-targets=generic -DOUTPUT"
mv main lavaMD-gpu && make clean
make CC=acpp VENDOR=AdaptiveCpp GPU=no  EXTRA_CFLAGS="--acpp-targets=generic -DOUTPUT"
mv main lavaMD-cpu   # cpu_selector variant, needed for llvmpipe

run() { ACPP_VISIBILITY_MASK="$2" ./$1 -boxes1d 10 && mv result.txt result.$3.txt; }
run lavaMD-gpu "omp;vk:MTL"      vk-anv
run lavaMD-gpu "omp;vk:NVIDIA"   vk-nvidia
run lavaMD-cpu "omp;vk:llvmpipe" vk-llvmpipe
run lavaMD-gpu "omp;ocl:Arc"     ocl-gpu
run lavaMD-cpu "omp;ocl:Ultra"   ocl-cpu
run lavaMD-cpu "omp"             omp
# reference: the same benchmark built with an AdaptiveCpp build that has the CUDA backend
ACPP_VISIBILITY_MASK="omp;cuda" ./lavaMD-gpu-cuda -boxes1d 10 && mv result.txt result.cuda.txt
```

Comparison (`compare.py result.X.txt result.cuda.txt`):

```python
import sys
def load(f): return [[float(v) for v in line.split(',')] for line in open(f)]
a, b = load(sys.argv[1]), load(sys.argv[2])
print(max(abs(x - y) / max(abs(y), 1e-6) for ra, rb in zip(a, b) for x, y in zip(ra, rb)))
```

Maximum relative difference against CUDA (100000 rows, 4 values each):

| Backend / device | max. rel. difference |
|---|---|
| OpenCL, Intel Arc Graphics (MTL) | 7.4e-6 |
| OpenCL, Intel Core Ultra 7 155H | 1.2e-5 |
| OpenMP | 1.3e-5 |
| **Vulkan, Intel Arc Graphics (MTL), ANV** | **42** |
| **Vulkan, NVIDIA RTX 500 Ada** | **21** |
| **Vulkan, llvmpipe** | **9.9** |

On a second machine (Radeon 780M) the Vulkan results had a maximum relative
error of 45 (RADV) and 9.9 (llvmpipe) against OpenMP, while HIP agreed with
OpenMP to 2e-5.

**Expected behavior**

The Vulkan results agree with the other backends to the same tolerance
(~1e-5).

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the Vulkan and OpenCL backends enabled; clspv built from source
  (upstream `a3d3e261` plus an unrelated local commit) following
  `doc/install-vulkan.md`; Vulkan SDK 1.4.341.1
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA: reference only (separate AdaptiveCpp build of the same commit, LLVM
  21.1.8, CUDA 13.4, driver 595.99.02)
* OpenCL: Intel compute-runtime 26.05.037020 (GPU), Intel OpenCL CPU runtime
  2026.21.3.0.31
* Vulkan drivers: Mesa 26.0.8 (ANV; llvmpipe with LLVM 21.1.8), NVIDIA 595.99.02
* Hardware/OS: Intel Core Ultra 7 155H with Intel Arc Graphics (Meteor Lake),
  NVIDIA RTX 500 Ada Generation Laptop GPU; Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Second machine: AMD Ryzen 7 PRO 8840HS, Radeon 780M, Mesa 25.2.8, Ubuntu 24.04
* Benchmark compiled with `acpp --acpp-targets=generic -O3` through its Makefile

**Optional additional diagnostic information**

The llvmpipe error (9.9) is identical on both machines. The program exits
normally on all devices and prints no AdaptiveCpp errors.

**Additional context**

The lavaMD kernel uses three `local_accessor`s, two of them of a struct type
(`FOUR_VECTOR`, four `float`s) and one of `float`
(`src/lavaMD-sycl/main.cpp`, lines 213–215 at the commit above). It has not been
reduced further.

---
