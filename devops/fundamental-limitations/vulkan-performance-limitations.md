# Performance of the AdaptiveCpp Vulkan backend: what to expect

This guide is for people running SYCL code on AdaptiveCpp's Vulkan backend.
It explains which performance limits come from Vulkan itself, which are
current limitations of AdaptiveCpp, and how to write code that runs well on
the backend today.

It is based on measurements on three GPUs and two CPUs, each compared with
AdaptiveCpp's native backend for the same hardware (October 2026, AdaptiveCpp
`develop` at `f4ff3681`):

| Hardware | Vulkan driver | Compared with |
|---|---|---|
| NVIDIA RTX 500 Ada Generation Laptop GPU | NVIDIA 595.99.02 | CUDA backend |
| Intel Arc Graphics (Core Ultra 7 155H, Meteor Lake) | Mesa ANV 26.0.8 | OpenCL backend (Intel compute-runtime) |
| AMD Radeon 780M (Ryzen 7 PRO 8840HS) | Mesa RADV 25.2.8 | HIP backend |
| Intel Core Ultra 7 155H CPU; AMD Ryzen 7 PRO 8840HS CPU | llvmpipe (Mesa) | OpenMP backend; Intel OpenCL CPU runtime |

All numbers come from
[`../vulkan-perf/findings.md`](../vulkan-perf/findings.md), which has the
programs, commands and raw results. Other GPUs and driver versions will
differ in detail.

## Summary

* **Large, simple kernels run as fast as on the native backends.** Streaming
  a large buffer reaches the same bandwidth, and long compute kernels with
  little synchronisation are within ±15%.
* **Every interaction with the host costs more.** Each kernel launch, copy
  and `wait()` costs tens of microseconds, and copies between host and device
  memory are much slower than natively. Programs with many small kernels,
  frequent synchronisation or frequent small copies lose the most.
* **Some code patterns are much slower than you would expect:** `sycl::fma`
  (40–73×), `size_t` division and modulo in index calculations (up to 15×),
  loops with runtime trip counts (up to 3.5×), and kernels with many
  local-memory barriers (up to 1.6×). Most have a workaround (see
  [Writing code for the Vulkan backend](#writing-code-for-the-vulkan-backend)),
  and all are tracked as improvement opportunities for AdaptiveCpp.
* **llvmpipe (Vulkan on the CPU) is for testing, not for speed.** Use the
  OpenMP backend for CPU execution.
* **If your GPU has a native AdaptiveCpp backend, it is usually faster
  today.** The Vulkan backend's strengths are portability and reaching GPUs
  that have no native backend.

## At a glance

Ratios are Vulkan time / native time (above 1: Vulkan is slower).

| Kind of work | NVIDIA: Vulkan vs CUDA | Intel Arc: Vulkan vs OpenCL | AMD 780M: Vulkan vs HIP |
|---|---|---|---|
| Streaming a large buffer (64–256 MB) | 113 vs 111 GB/s | 63 vs 64 GB/s | 66–73 vs 66–70 GB/s |
| Long, simple kernels (gaussian, the black-scholes kernel, atomicReduction) | 0.99–1.15× | 1.00–1.02× | 0.88–1.04× |
| Kernels with barriers, heavy maths or index arithmetic (pathfinder, binomial, nbody, su3, mandelbrot) | 1.3–3.7× | 1.0–3.5× | 0.47–2.6× |
| Empty kernel launch + `wait()` | 37 vs 10 µs | 81 vs 72 µs | 88 vs 20 µs |
| Small host↔device copy + `wait()` | 0.57–0.74 ms vs ~0.01 ms | 0.14–0.24 ms vs ~0.1 ms | 0.15–0.18 ms vs ~0.03 ms |
| Large host↔device copies | 2.1–3.0 vs 8.1–8.4 GB/s | to device 3.3–3.6 vs 11–13 GB/s; **from device 0.16** vs 11–13 GB/s | 9–53× slower at 1–2 MB |

## Fundamental limitations

These follow from Vulkan, its SPIR-V environment or the drivers. AdaptiveCpp
can reduce their cost, but they will not disappear.

### 1. Every submission and every synchronisation has a fixed cost

Each operation submitted to a Vulkan queue goes through the driver (and, on
most drivers, the operating-system kernel), and AdaptiveCpp orders dependent
operations with timeline semaphores that the driver resolves. Native backends
such as CUDA and HIP can order work inside the GPU's own queue. The result is
a fixed cost per operation and per host synchronisation:

| Empty kernel | Vulkan | Native |
|---|---|---|
| NVIDIA, back-to-back launches | 17.9 µs each | CUDA 5.2 µs |
| NVIDIA, launch + `wait()` | 37 µs | CUDA 10 µs |
| AMD 780M, back-to-back launches | 50 µs each | HIP 11 µs |
| AMD 780M, launch + `wait()` | 88 µs | HIP 20 µs |
| Intel Arc, back-to-back launches | 33.7 µs each | OpenCL 41.6 µs |
| Intel Arc, launch + `wait()` | 81 µs | OpenCL 72 µs |

Part of this is AdaptiveCpp's current one-submission-per-operation design,
which can be improved by batching; the cost of each host round trip stays.
On the Intel GPU the native OpenCL stack pays about the same.

**Effect:** floydwarshall (1024 dependent launches per run) is 1.27× slower
than CUDA and 1.37× slower than HIP. A breadth-first search on a deep graph
(1999 levels, each with two kernel launches and a `wait()`) spends 2.0–2.7×
longer than natively in those launches.

**What to do:**
* Prefer fewer, larger kernels; fuse small kernels. Kernels shorter than
  ~100 µs lose a large part of their time to these costs.
* Don't call `wait()` inside loops unless you must. If a loop tests a
  convergence flag on the host, consider testing it every few iterations
  instead of every iteration.

Source: [FUND-1](../vulkan-perf/findings.md#fund-1), [IMPR-1](../vulkan-perf/findings.md#impr-1), [IMPR-2](../vulkan-perf/findings.md#impr-2).

### 2. Host data always moves through explicit, staged copies

The Vulkan backend has no host or shared USM: `sycl::malloc_host` and
`sycl::malloc_shared` throw `errc::feature_not_supported`, because a pointer
returned by `vkMapMemory` cannot be used as a device pointer in a kernel. Data
used by kernels lives in device USM, and every copy between host memory and
device memory goes through an intermediate mapped buffer, with extra
synchronisation. How expensive that is today is mostly a current limitation
(see the [table below](#current-limitations-and-workarounds)): 0.14–0.74 ms
per small copy and 2–4 GB/s for large ones.

**What to do:** keep data on the device for as long as possible, copy it in
large blocks, and combine many small copies into one.

Source: [FUND-4](../vulkan-perf/findings.md#fund-4), [IMPR-3](../vulkan-perf/findings.md#impr-3).

### 3. Correctly rounded maths may need software routines

Vulkan guarantees less precision for some operations than SYCL requires.
SYCL's `sycl::fma` must be correctly rounded, but Vulkan's standard `Fma`
instruction only has to be as precise as a separate multiply and add. Only
devices with the `VK_KHR_shader_fma` extension provide a correctly rounded
FMA instruction: the NVIDIA driver here has it; Mesa 26.0.8's ANV and llvmpipe
do not. Without it, `sycl::fma` must be computed in software, and today
AdaptiveCpp does this on every device:

| 1024 dependent steps per work-item | `x*a + b` | `sycl::fma(x, a, b)` |
|---|---|---|
| Vulkan, NVIDIA | 6.8 ms | **499 ms** |
| CUDA, same GPU | 2.8 ms | 2.8 ms |
| Vulkan, Intel Arc | 8.5 ms | **351 ms** |
| OpenCL, same GPU | 4.8 ms | 4.4 ms |
| Vulkan, llvmpipe | 0.53 s | **5.3 s** |

On devices with the extension this can be fixed in AdaptiveCpp (a quick
experiment brought the NVIDIA case down to 4.5 ms); on the others it is
inherent.

**What to do:** write `a*b + c` instead of `sycl::fma(a, b, c)` unless you
need the correctly rounded result. `sycl::mad(a, b, c)` produces the same
code as `a*b + c`.

Source: [FUND-6](../vulkan-perf/findings.md#fund-6), [IMPR-16](../vulkan-perf/findings.md#impr-16).

### 4. Memory accesses use 64-bit pointers that drivers know little about

Kernels access device memory through 64-bit device addresses (Vulkan
"physical storage buffers"), which carry no information about aliasing or
read-only use. Two consequences were observed in the generated machine code:

* **Uniform loads.** When every work-item of a group reads the same address
  (e.g. `p[j]` in an all-pairs loop), the native AMD and Intel compilers use
  scalar or single-lane loads. Through Vulkan, RADV and ANV issue a normal
  per-work-item load instead. This is part of why nbody is 1.23× slower than
  HIP on the AMD GPU and 2.48× slower than OpenCL on the Intel GPU.
* **64-bit address arithmetic on Intel GPUs.** Meteor Lake's GPU has no 64-bit
  integer ALU, so every 64-bit address calculation is emulated with several
  32-bit instructions; Intel's OpenCL compiler uses 32-bit buffer offsets
  instead. In nbody's inner loop on ANV, integer instructions outnumber
  floating-point ones by more than two to one. Part of this is index
  arithmetic that AdaptiveCpp could do in 32 bits; the address arithmetic
  itself stays 64-bit.

**What to do:**
* Use 32-bit index types (`unsigned`, `int`) for index arithmetic when sizes
  allow it (see the table below).
* Avoid serial loops in a `single_task` (for example to sum an array); use
  `sycl::reduction` or group algorithms.

Source: [FUND-2](../vulkan-perf/findings.md#fund-2), [IMPR-6](../vulkan-perf/findings.md#impr-6).

### 5. llvmpipe is a testing device, not a fast CPU backend

llvmpipe runs Vulkan compute on the CPU. Every buffer access becomes per-lane
gathers and scatters instead of contiguous vector loads and stores, so
streaming kernels reach 27–31 GB/s against 40–47 GB/s for OpenMP and the Intel
OpenCL CPU runtime (16–17 vs 27–32 GB/s on the AMD laptop). On real kernels
llvmpipe was 1.2–4.9× slower than OpenMP on most benchmarks (faster on
pathfinder, nbody and su3) and 1.4–11× slower than the Intel OpenCL CPU
runtime on most benchmarks.

**What to do:** use the OpenMP backend (or an OpenCL CPU runtime) to run on
the CPU; use llvmpipe to test Vulkan code paths on machines without a GPU.

Source: [FUND-3](../vulkan-perf/findings.md#fund-3), [FUND-5](../vulkan-perf/findings.md#fund-5).

## Current limitations and workarounds

These are limitations of AdaptiveCpp's Vulkan backend at the time of writing.
Each can be fixed in AdaptiveCpp; the fixes are tracked in
[`../performance-improvements/vulkan-performance-opportunities.md`](../performance-improvements/vulkan-performance-opportunities.md).

| Limitation | Measured cost | What to do today | Tracked as |
|---|---|---|---|
| Each host↔device copy allocates and maps a new staging buffer | 0.57–0.74 ms per small copy on NVIDIA (CUDA ~0.01 ms); 2–4 GB/s for large copies | keep data on the device; combine small copies; avoid copies inside loops | IMPR-3 |
| Copies from the device read uncached memory on Intel GPUs (ANV) | 0.16 GB/s (OpenCL 13 GB/s): 256 MB take 1.7 s | avoid large device-to-host copies, or use the OpenCL backend for such programs on Intel GPUs | IMPR-13 |
| Index arithmetic on `size_t` is 64-bit, and division or modulo by a constant is not optimised | index decomposition (`i / 36`, `i % 9`, ...): 15× slower than with `unsigned` on NVIDIA, 3.9× on Intel | use 32-bit index types for such arithmetic | IMPR-6 |
| `sycl::fma` is a software routine | 40–73× slower than `a*b + c` | write `a*b + c` unless you need the correctly rounded result | IMPR-16 |
| `a*b + c` is never fused into an FMA | a chain of `x*a + b` is 2.5× slower than with CUDA on NVIDIA; fusing would save 15% in nbody | nothing: `-ffast-math` and `sycl::mad` make no difference | IMPR-5 |
| `sycl::rsqrt` is a multi-instruction sequence | nbody kernel ~1.17× slower than with `sycl::native::rsqrt` (NVIDIA and Intel) | use `sycl::native::rsqrt` if its precision is enough | IMPR-11 |
| `sycl::native::divide` is broken: the kernel fails to load | — | use `a * sycl::native::recip(b)` (1.5–1.6× faster than `a / b` on both GPUs) or `a / b`; plain division is not slower than natively | BUG-15 |
| Loops with a runtime trip count are not unrolled | a serial loop is 3.5× slower than CUDA on NVIDIA | avoid serial loops; if you need one, see [Unrolling](#unrolling-only-on-the-vulkan-backend) | IMPR-12 |
| Each `malloc_device` / `free` is a new driver allocation | Intel: the first kernel on a fresh 8 MB buffer takes twice as long; NVIDIA: a driver allocation costs ~0.1 ms (plain Vulkan measurement) | allocate once and reuse buffers across iterations | IMPR-15 |
| Every work-group barrier also orders global memory | pathfinder 1.6× and binomial 1.4× slower on NVIDIA than with local-only barriers; ~1.1× on Intel | nothing | IMPR-14 |
| One queue submission per operation | see [limitation 1](#1-every-submission-and-every-synchronisation-has-a-fixed-cost) | fuse kernels, avoid `wait()` in loops | IMPR-1, IMPR-2 |
| Histogram-style atomics are slower | 1.1–3.3× | none known | IMPR-10 |

## Writing code for the Vulkan backend

A checklist, roughly in order of impact:

1. **Keep data on the device**, copy in large blocks, and avoid small copies
   and `wait()` inside loops.
2. **Fuse small kernels**; aim for kernels that run for at least ~100 µs.
3. **Allocate device memory once** and reuse it.
4. **Use 32-bit index types** (`unsigned`, `int`) for index arithmetic,
   especially division and modulo, when sizes fit in 32 bits.
5. **Don't use `sycl::fma` in hot code**; write `a*b + c`.
6. **Use `sycl::native::rsqrt` and `a * sycl::native::recip(b)`** where their
   precision is acceptable. Don't use `sycl::native::divide` on Vulkan.
7. **Avoid serial loops** such as `single_task` reductions; use
   `sycl::reduction` or group algorithms.
8. **Don't expect `-ffast-math` to help** on the Vulkan backend: it made no
   measurable difference.
9. **Use device USM** (`sycl::malloc_device`); host and shared USM are not
   available.

### Unrolling only on the Vulkan backend

`#pragma unroll 8` on a serial loop with a runtime trip count made it 2.5–3.7×
faster on all three Vulkan drivers, but made the same loop 2.7× slower with
the CUDA backend. With AdaptiveCpp's default generic JIT compiler
(`--acpp-targets=generic`), JIT-time reflection
([`ACPP_EXT_JIT_COMPILE_IF`](../../doc/extensions.md)) can restrict it to the
Vulkan backend at no run-time cost:

```cpp
namespace jit = sycl::AdaptiveCpp_jit;

inline float sum_unrolled(const float *in, size_t n) {
  float s = 0;
#pragma unroll 8
  for (size_t i = 0; i < n; i++) s += in[i];
  return s;
}

inline float sum_plain(const float *in, size_t n) {
  float s = 0;
  for (size_t i = 0; i < n; i++) s += in[i];
  return s;
}

// inside the kernel:
__acpp_if_target_sscp(
  jit::compile_if_else(
    jit::reflect<jit::reflection_query::runtime_backend>() ==
        static_cast<int>(sycl::backend::vk),
    [&]() { out[0] = sum_unrolled(in, n); },
    [&]() { out[0] = sum_plain(in, n); });
);
```

With this, the loop took 1.0–1.1 ms on Vulkan (NVIDIA; 3.15 ms without the
pragma) while CUDA kept its 0.89 ms. The `#pragma` has to be outside the
`__acpp_if_target_sscp(...)` macro argument, hence the helper functions.

## Measuring performance on the Vulkan backend

* **Time with a wall clock** around `wait()`. SYCL event profiling reports
  kernel durations that are 34–44× too short on Intel GPUs and 10× too short
  on AMD GPUs ([BUG-1](../bug-reports/BUG-1-vk-profiling-timestamp-period.md));
  it is correct on NVIDIA and llvmpipe.
* **With `enable_profiling`, keep and wait on every event.** Dropping events
  of a profiling queue can crash llvmpipe and lose the device on AMD
  ([BUG-2](../bug-reports/BUG-2-vk-query-pool-lifetime.md)).
* **Discard the first run** after building or changing a program: kernels are
  JIT-compiled (AdaptiveCpp prints a `kernel_cache` warning when that
  happened).
* **Check which device ran.** Select devices with e.g.
  `ACPP_VISIBILITY_MASK="omp;vk:NVIDIA"`, and print the device name: a
  misspelt backend name makes every backend visible, and a device name that
  matches nothing leaves no device, both silently
  ([BUG-3](../bug-reports/BUG-3-visibility-mask-invalid-entries.md)).
* **On laptops**, run on AC power with a performance power profile, and repeat
  runs: integrated GPU clocks vary by up to 2× between runs.
* **Verify results** against another backend. At the time of writing these
  produce wrong results or fail on the Vulkan backend: HeCBench lavaMD and lud
  on every driver ([BUG-8](../bug-reports/BUG-8-vk-lavamd-wrong-results.md),
  [BUG-9](../bug-reports/BUG-9-vk-lud-wrong-results.md)), sub-group
  operations on Intel GPUs
  ([BUG-10](../bug-reports/BUG-10-vk-anv-subgroup-size.md)), FP64 atomics on
  devices without them
  ([BUG-6](../bug-reports/BUG-6-vk-unsupported-spirv-capabilities.md)),
  `char` local-memory accesses at unaligned offsets, as in scan kernels
  ([BUG-11](../bug-reports/BUG-11-vk-clspv-unaligned-local-gep.md)), and
  `sycl::native::divide`
  ([BUG-15](../bug-reports/BUG-15-vk-native-divide.md)).

## Choosing a backend

| Hardware | Recommendation today |
|---|---|
| NVIDIA GPU | **CUDA.** Vulkan matches it for streaming and for gaussian, softmax and atomicReduction, but is 1.1–3.7× slower on the other kernels measured and much slower for copies. |
| Intel GPU | **OpenCL.** Vulkan matches it on floydwarshall, gaussian, the black-scholes kernel, atomicReduction and su3, but is 1.9–3.5× slower on nbody, mandelbrot, pathfinder and binomial, copies from the device are 80× slower, and sub-group operations currently give wrong results. (Level Zero was not measured.) |
| AMD GPU (RDNA3 APU measured) | **Either.** Vulkan (RADV) matched or beat HIP on compute-heavy kernels with few launches (pathfinder 0.47×, softmax 0.76×, black-scholes 0.88×, binomial 0.91×, gaussian 0.92×), but was slower with many launches (1.37×), atomics (1.4×), 64-bit index arithmetic (su3 2.6×) and copies. |
| GPU without a native AdaptiveCpp backend | **Vulkan**, following the advice above. |
| CPU | **OpenMP** (or an OpenCL CPU runtime). llvmpipe only for testing. |
