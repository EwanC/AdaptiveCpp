# BUG-3: Invalid `ACPP_VISIBILITY_MASK` makes every backend visible

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`); could also be filed as a feature request |
| Suggested title | `ACPP_VISIBILITY_MASK: an invalid entry discards the whole mask, and filters that match no device are silent` |
| Backend / devices | Runtime (all backends) |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-3 (disputed), run R13 (M2) |
| Note | The original finding ("`omp;vulkan` is silently ignored") does **not** reproduce: `vulkan` is an accepted alias for `vk` since `98630f7a`. This report covers the behaviour that remains. Low priority. |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

Two cases make it easy to benchmark or test a different device than intended:

1. If `ACPP_VISIBILITY_MASK` contains an unknown backend name, the runtime
   prints a message and then ignores the *entire* variable, so all backends
   and devices become visible, including the ones the valid entries were meant
   to exclude.
2. A device filter that matches no device (e.g. a typo in a device name) leaves
   that backend without devices and prints nothing.

**To Reproduce**

```sh
ACPP_VISIBILITY_MASK="omp;vulkn" acpp-info -l
ACPP_VISIBILITY_MASK="omp;vk:NoSuchDevice" acpp-info -l
```

Output (OpenCL, OpenMP and Vulkan backends installed):

```
$ ACPP_VISIBILITY_MASK="omp;vulkn" acpp-info -l
'vulkn' is not a known backend name.
AdaptiveCpp settings parsing: Could not parse value of environment variable: ACPP_VISIBILITY_MASK
=================Backend information===================
Loaded backend 0: OpenCL
  Found device: Intel(R) Arc(TM) Graphics
  Found device: Intel(R) Core(TM) Ultra 7 155H
Loaded backend 1: OpenMP
  Found device: AdaptiveCpp OpenMP host device
Loaded backend 2: Vulkan
  Found device: Intel(R) Arc(tm) Graphics (MTL)
  Found device: NVIDIA RTX 500 Ada Generation Laptop GPU
  Found device: llvmpipe (LLVM 21.1.8, 256 bits)

$ ACPP_VISIBILITY_MASK="omp;vk:NoSuchDevice" acpp-info -l
=================Backend information===================
Loaded backend 0: OpenMP
  Found device: AdaptiveCpp OpenMP host device
Loaded backend 1: Vulkan
  (no devices found)
```

For comparison, valid masks behave as documented:

```
$ ACPP_VISIBILITY_MASK="omp;vulkan:llvmpipe" acpp-info -l
=================Backend information===================
Loaded backend 0: OpenMP
  Found device: AdaptiveCpp OpenMP host device
Loaded backend 1: Vulkan
  Found device: llvmpipe (LLVM 21.1.8, 256 bits)
```

An application using `sycl::default_selector_v` with the misspelt mask above
runs on whichever device the default selector prefers among all backends
instead of failing.

**Expected behavior**

An invalid `ACPP_VISIBILITY_MASK` should not widen device visibility: either
treat it as a fatal configuration error, or apply the valid entries and warn
about the invalid one. A device filter that matches no device should produce
a warning.

**Describe your setup**

* AdaptiveCpp built from source (CMake `Release`) from `develop` at
  `f4ff3681` ("[VK] Create pass to legalize int widths for clspv (#2272)"),
  with the Vulkan and OpenCL backends enabled
* clang: Ubuntu clang 20.1.8 (`/usr/lib/llvm-20`, distribution packages)
* Host compiler: GCC 15.2.0
* CUDA / ROCm: not involved
* OpenCL: Intel compute-runtime 26.05.037020 (GPU), Intel OpenCL CPU runtime
  2026.21.3.0.31 (only listed by `acpp-info`)
* Vulkan drivers: Mesa 26.0.8 (ANV; llvmpipe with LLVM 21.1.8), NVIDIA 595.99.02
* Ubuntu 26.04.1, Linux 7.0.0-38-generic
* No application compilation needed; `acpp-info` from the installation is enough

**Optional additional diagnostic information**

None beyond the `acpp-info` output above.

**Additional context**

Relevant code at `f4ff3681`: `operator>>(std::istream&, visibility_mask_t&)`
in `src/runtime/settings.cpp` sets `failbit` and stops parsing on an unknown
backend name; the settings parser then reports
`Could not parse value of environment variable` and the mask is not applied.
`device_matches()` in the same file returns `false` for every device when no
filter matches, without a diagnostic.

---
