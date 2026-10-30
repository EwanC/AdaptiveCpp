# Vulkan backend performance analysis

Living document tracking performance of the AdaptiveCpp Vulkan backend
compared with other backends targeting the *same* hardware:

* Vulkan GPU driver (e.g. RADV, ANV, NVIDIA) vs the native GPU backend
  (HIP / CUDA / OpenCL / L0)
* Vulkan CPU driver (llvmpipe/lavapipe) vs the OpenMP backend and, where
  available, an OpenCL CPU runtime

The document is designed to be extended from multiple machines. Every finding
has a stable ID; new benchmark runs are appended to the [run log](#run-log)
and then referenced from the finding's **Evidence** table with a verdict.

Documents derived from this one (keep them in sync when a finding changes):

* [`../bug-reports/`](../bug-reports/README.md): one upstream-ready issue
  draft per bug (BUG-n).
* [`../performance-improvements/`](../performance-improvements/vulkan-performance-opportunities.md):
  every improvement opportunity (IMPR-n) with evidence, proposal and how to
  validate it.
* [`../fundamental-limitations/`](../fundamental-limitations/vulkan-performance-limitations.md):
  user-facing guide to what performance to expect and how to write code
  for the Vulkan backend.

## How to contribute a run

1. Add a machine entry to [Machines](#machines) (`M<n>`) if it isn't there.
2. Add a run entry to the [Run log](#run-log) (`R<n>`) with the commands and
   raw numbers.
3. For every finding the run touches, add a row to that finding's
   **Evidence** table with a verdict:
   * **confirms** – reproduces the finding
   * **refutes** – contradicts it (explain why in Notes)
   * **partial** – reproduces with a different magnitude / only some aspects
   * **n/a** – the finding doesn't apply to this hardware/driver
4. If a run reveals something new, add a finding with the next free ID in the
   right section. Never renumber or delete IDs; mark them
   `Status: withdrawn` instead.
5. Update `Status` when warranted: `open`, `confirmed` (≥2 machines),
   `disputed`, `fixed (<commit/PR>)`, `withdrawn`.

### Measurement rules (learned the hard way)

* `ACPP_VISIBILITY_MASK` backend key is **`vk`**, not `vulkan` (on M2's
  build `vulkan` is accepted as an alias, see [BUG-3](#bug-3)). Unknown keys
  are silently ignored, so a typo silently benchmarks another device. Always
  confirm the device (e.g. `ACPP_DEBUG_LEVEL=3` or print the device name).
  Device selection by name: `omp;vk:RADV`, `omp;vk:llvmpipe`.
* Discard the first run after a build (JIT compile). The runtime prints a
  `kernel_cache` warning when new binaries were JIT-compiled.
* Do **not** use SYCL event profiling on the Vulkan backend until
  [BUG-1](#bug-1) is fixed; use wall-clock timing around `wait()`.
* Do not run with an `enable_profiling` queue and dropped events (crash,
  [BUG-2](#bug-2)).
* iGPU clocks ramp; repeat runs 2-3× and report the steady state.
* Device names in a mask are substrings, e.g. `omp;vk:MTL`, `omp;vk:NVIDIA`,
  `omp;ocl:Arc`, `omp;ocl:Ultra` on M2. The mask is read up to the first
  whitespace, so pick a substring without spaces. `cpu_selector` prefers a
  non-OpenMP CPU device, so `omp;vk:llvmpipe` / `omp;ocl:<cpu>` select it.
* A misspelt backend name prints `'x' is not a known backend name`, but
  then the *whole* mask is ignored and all backends load; a device filter
  that matches nothing leaves the backend without devices and prints
  nothing ([BUG-3](#bug-3)).
* When the native backend lives in a different AdaptiveCpp build (CUDA on
  M2), build that variant with that build's `acpp`
  (`hecbench_build.sh` with `ACPP_CUDA`) and note that its LLVM version may
  differ from the Vulkan build's.
* SYCL event profiling also aborts on the OpenCL backend on M2
  (`Operation not profiled: No execution end timestamp available`; the
  backend has no profiling support but accepts `enable_profiling`,
  [BUG-12](#bug-12)), so wall-clock timing is the only method that works
  across all backends.
* The MTL iGPU's clock state can change a kernel's time by up to 2× between
  batches (e.g. right after using the other GPU): interleave A/B runs,
  discard the first run of each batch and compare steady states.
* For A/B experiments that change generated code (clspv flags via
  `clspv_shim.c`, `patch_barrier.sh`) use a private `ACPP_APPDB_DIR`, so the
  JIT cache is rebuilt and the default `~/.acpp` cache is left untouched.

### Tools

Sources in [`tools/`](tools/). Build each with
`acpp -O3 --acpp-targets=generic <file>.cpp -o <name>` (when the native
backend needs another AdaptiveCpp build, also build `<name>-cuda` etc. with
that build's `acpp`; `run_tool.sh` picks it up via a binary suffix).

| Tool | Purpose |
|------|---------|
| `mb.cpp [n]` | Launch latency, small memcpy latency, nbody kernels timed individually (wall clock) |
| `bw.cpp <MB>` | STREAM-like `b[i]=2*a[i]+1` bandwidth & per-kernel cost for back-to-back dependent kernels |
| `lat.cpp <K> <M>` | K empty kernel launches, then M 4-byte D2H copies; target for `strace -f -c` and sampling |
| `samp.sh` / `samp.gdb` / `agg.py` | gdb-based stack sampler for when `perf` / ptrace-attach is unavailable |
| `hecbench_run.sh` | Build instructions + runner for a HeCBench SYCL benchmark on all comparison devices (warm-up + measured run) |
| `xfer.cpp [max_MB]` | Host↔device `memcpy`+`wait()` sweep from 4 B to 256 MB with pageable host memory (latency and bandwidth) |
| `idx.cpp [n_M] [iters]` | su3-style index decomposition (`t/36`, `(t%36)/9`, ...) in `size_t` vs `unsigned`; isolates IMPR-6 |
| `prof.cpp [MB]` | SYCL event profiling vs wall clock for one kernel (BUG-1); its 64-step `x*a+b` chain also exposes IMPR-5 |
| `rsq.cpp [n]` | nbody acceleration kernel with `sycl::rsqrt`, `sycl::native::rsqrt` and `1/sycl::sqrt` (IMPR-11) |
| `mandel.cpp [iters]` | HeCBench mandelbrot kernel with a reused vs a freshly allocated output buffer (IMPR-15) |
| `bug2.cpp` | BUG-2 reproducer (profiling queue, dropped events); run with the validation layer |
| `memtype.cpp [MB]` | Plain Vulkan (`g++ -O2 memtype.cpp -I$VULKAN_SDK/include -L$VULKAN_SDK/lib -lvulkan`): host read/write bandwidth and allocate+map cost of every `HOST_VISIBLE` memory type |
| `run_tool.sh` / `tool_median.py` | Run a tool on every device (warm-up + `REPS` runs) / median table for `label: value` outputs such as `mb` |
| `gen_bfs_graph.py` | Rodinia-format graphs for `bfs`: `random` (like `graph1MW_6`) or `grid` (deep BFS, 2·side−1 levels) |
| `hecbench_build.sh` / `hecbench_suite.sh` / `hecbench_parse.py` | Build gpu/cpu(/gpu-cuda) variants; run the recorded benchmark set and arguments; extract metrics and Vulkan/native ratios |
| `clspv_shim.c` + `clspv_wrapper.sh` | `LD_PRELOAD` shim that re-runs AdaptiveCpp's clspv invocation with extra flags (`CLSPV_EXTRA`) without rebuilding anything |
| `patch_barrier.sh` | A/B: rewrite the `OpControlBarrier` memory semantics in an app's cached clspv SPIR-V (IMPR-14) |
| `uguide.cpp` | Checks for user-level workarounds: `x*a+b` vs `sycl::fma` chain, `a/x` vs `a*native::recip(x)` chain, serial loop plain vs `#pragma unroll 8` vs manual ×8 (IMPR-5, IMPR-11, IMPR-12, IMPR-16) |
| `jit_unroll.cpp` | The serial loop with `#pragma unroll 8` applied only on the Vulkan backend through JIT reflection (`compile_if_else` on `reflection_query::runtime_backend`) |

### Correctness gate for HeCBench data

Only record performance data for a benchmark/device pair if the result is
verified correct. Many HeCBench benchmarks only self-check optionally (e.g.
`lud -v`) or not at all (e.g. `pathfinder` needs `-DBENCH_PRINT`, `lavaMD`
needs `-DOUTPUT`); cross-check their outputs between devices before trusting
timings. Failures go in [BUG-5](#bug-5) without performance data.

Useful external tools: `RADV_DEBUG=shaders,nocache` (dumps ACO ISA),
`spirv-dis` on JIT cache entries in `~/.acpp/apps/<app>/jit-cache/*.jit`,
`llvm-objdump -d --mcpu=<gfx>` on HIP JIT cache ELFs,
`ACPP_S2_DUMP_IR_FINAL=1` (dumps the LLVM IR handed to clspv, as
`<source>.ll` in the working directory),
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`.
On Intel/NVIDIA (M2): `MESA_SHADER_CACHE_DISABLE=true INTEL_DEBUG=cs` (ANV
ISA and SIMD width), `NEO_CACHE_PERSISTENT=0 IGC_ShaderDumpEnable=1
IGC_DumpToCustomDir=<dir>` (IGC ISA for the OpenCL backend),
`GALLIVM_DEBUG=asm` (llvmpipe x86 code; works in Ubuntu's release Mesa), and
the PTX the CUDA backend JIT-compiled is stored as text in its JIT cache.
`perf` is often unavailable on laptops (`perf_event_paranoid=4`); `samp.sh`
still works because gdb starts the program as its child.

---

## Machines

### M1
* CPU/GPU: AMD Ryzen 7 PRO 8840HS, Radeon 780M (gfx1103, RDNA3 APU), 16 threads
* OS: Linux 6.17.0-1028-oem (Ubuntu 24.04)
* Vulkan: Mesa 25.2.8 (RADV PHOENIX; llvmpipe LLVM 20.1.2, 256-bit), Vulkan SDK 1.4.321.1
* RADV `timestampPeriod` = 10, subgroup size 64; llvmpipe `timestampPeriod` = 1, subgroup size 8
* Memory: every RADV device allocation lands in a `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` type (APU)
* AdaptiveCpp: `f4ff3681` (Release build), clspv per `doc/install-vulkan.md`

### M2
* Lenovo ThinkPad P16s Gen 3: Intel Core Ultra 7 155H (Meteor Lake-H,
  6P+8E+2LP-E cores, 22 threads, 24 MB L3), 32 GB RAM, on AC with the
  `performance` power profile (`powerprofilesctl`; the default
  `power-saver` throttles CPU and both GPUs)
* iGPU: Intel Arc Graphics (MTL, Xe-LPG, 128 EU, 0x7d55, i915, 0.8–2.25 GHz)
* dGPU: NVIDIA RTX 500 Ada Generation Laptop GPU (AD107, 16 SM, 4 GB,
  PCIe 4.0 x8)
* OS: Ubuntu 26.04.1, Linux 7.0.0-38-generic
* Vulkan: Mesa 26.0.8 (ANV; llvmpipe LLVM 21.1.8, 256-bit), NVIDIA 595.99.02,
  Vulkan SDK 1.4.341.1
* OpenCL: Intel compute-runtime (NEO) 26.05.037020 for the iGPU, Intel CPU
  runtime 2026.21.3.0.31 (PoCL is installed but hidden by AdaptiveCpp as it
  lacks SPIR-V)
* CUDA: driver 595.99.02, toolkit 13.4
* `timestampPeriod`: ANV 52.083, NVIDIA 1, llvmpipe 1. Sub-group size: ANV
  reports 32 (supports 8–32, compiles SIMD16 unless a size is required),
  NVIDIA 32, llvmpipe 8
* Memory types: ANV: every type is `DEVICE_LOCAL|HOST_VISIBLE`; the
  `HOST_COHERENT` types are uncached (write-combined) and the `HOST_CACHED`
  types are not coherent. NVIDIA: `DEVICE_LOCAL`;
  `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` (ReBAR, 4 GB, used for device USM);
  `HOST_VISIBLE|HOST_COHERENT[|HOST_CACHED]` system memory (used for staging).
  llvmpipe: one type with all flags
* AdaptiveCpp `9eb12573` (= `f4ff3681` + this directory), two Release builds:
  LLVM 20.1.8 with OpenCL + Vulkan + OpenMP (clspv `2ecc89b5`) for everything
  except CUDA, and LLVM 21.1.8 with CUDA + OpenCL + OpenMP for the CUDA
  binaries
* HeCBench `8d66934f`
* Comparisons: Vulkan ANV vs OpenCL (NEO) on the iGPU, Vulkan NVIDIA vs CUDA
  on the dGPU, llvmpipe vs the Intel OpenCL CPU runtime (OpenMP as a second
  CPU reference). Masks: `omp;vk:MTL`, `omp;ocl:Arc`, `omp;vk:NVIDIA`,
  `omp;cuda`, `omp;vk:llvmpipe`, `omp;ocl:Ultra`, `omp`

---

## Run log

### <a id="r1"></a>R1 — M1, nbody-sycl (HeCBench), 2026-10-29
Binaries: `nbody-gpu` (gpu_selector), `nbody-cpu` (cpu_selector),
built with `--acpp-targets=generic`. `./nbody-<x> 16000 30`, warm cache, GFLOP/s:

| Device | Mask | GFLOP/s | Total s |
|--------|------|---------|---------|
| 780M HIP | `omp;hip` | 850 ± 135 | 0.206 |
| 780M Vulkan RADV | `omp;vk:RADV` | 690 ± 90 | 0.255 |
| 780M Vulkan RADV, clspv `--cl-mad-enable` (manually swapped JIT cache) | `omp;vk:RADV` | 765 ± 107 | 0.233 |
| CPU OpenMP | `omp` | 41 ± 2 | 4.02 |
| CPU llvmpipe | `omp;vk:llvmpipe` | 116 ± 6 | 1.40 |
| CPU llvmpipe, `--cl-mad-enable` | `omp;vk:llvmpipe` | ~121 | — |

nbody reports `FAIL` on RADV at 30 steps; energies are identical to HIP
(172663). The benchmark uses an absolute tolerance of 1e-3 on a value of
~1.7e5, which is below float ULP — benchmark artefact, not a Vulkan bug.

### <a id="r2"></a>R2 — M1, `tools/mb.cpp` (n=16000), 2026-10-29
Wall clock, steady state:

| Metric | HIP | RADV | OpenMP | llvmpipe |
|--------|-----|------|--------|----------|
| Empty kernel, host submit cost (pipelined, 500×) | 7 µs | 48 µs | 1.4 µs | 35 µs |
| Empty kernel, end-to-end pipelined | 11 µs | 50 µs | 5.7 µs | 37 µs |
| Empty kernel + `wait()` | 20 µs | 88 µs | 11 µs | 51 µs |
| D2H 4 B memcpy + wait | 34 µs | 151 µs | 16 µs | 33 µs |
| H2D 4 B memcpy + wait | 26 µs | 177 µs | 9 µs | 33 µs |
| H2D 625 KB | 19 ms (pageable) | 0.64 ms | 0.08 ms | 0.84 ms |
| nbody accel kernel + wait | 8.4 ms | 8.5 ms | 117 ms | 46 ms |
| nbody update kernel + wait | 63 µs | 235 µs | 62 µs | 158 µs |
| `single_task` serial sum of 16000 floats + wait | 1.17 ms | 1.68 ms | 0.06 ms | 0.11 ms |

### <a id="r3"></a>R3 — M1, `tools/bw.cpp`, 2026-10-29
µs per kernel for 50 back-to-back dependent kernels (no waits between):

| Buffer | HIP | RADV | OpenMP | llvmpipe |
|--------|-----|------|--------|----------|
| tiny (launch only) | 10 | 47 | — | 11 |
| 1 MB | 53 | 89 | — | 133 |
| 2 MB | 102 | 137 | 14 | 210 |
| 4 MB | 161 | 196 | 24 | 373 |
| 8 MB | 253 | 304 | 142 | 730 |
| 16 MB | 511 (66 GB/s) | 511 (66 GB/s) | — | — |
| 64 MB | — | — | 5050 (27 GB/s) | 8088 (17 GB/s) |
| 128 MB | 3858 (70 GB/s) | 3682 (73 GB/s) | — | — |
| 256 MB | — | — | 16918 (32 GB/s) | 32794 (16 GB/s) |

llvmpipe 64 MB with `LP_NUM_THREADS`=4/8/16: 10.0/16.6/16.4 GB/s.
OpenMP with `OMP_NUM_THREADS=8`: 33.9 GB/s.

### <a id="r4"></a>R4 — M1, syscalls / sampling, 2026-10-29
`strace -f -c ./lat K M` (net of the K=M=0 baseline):
* RADV, 10000 launches: ~3 ioctl per launch.
* RADV, 1000 4-byte D2H copies: ~11 ioctl + 1.4 mmap + 1.1 munmap per copy.
* llvmpipe, 10000 launches: ~57 futex per launch.

gdb sampling of the submitting thread (`samp.sh`, ~200 samples):
* RADV launch loop: 90% in ioctl; 56% `vk_queue::submit_command_buffer`
  (vkQueueSubmit), 22% `vk_queue::get_command_buffer` → `drmSyncobjQuery2`
  (timeline `getCounterValue`), 17% `queue::wait`.
* RADV memcpy loop: 69% `event::wait` (`drmSyncobjTimelineWait`), 8%
  `vk_allocator::staging_allocate` (`amdgpu_bo_alloc`), 14% submit.
* llvmpipe launch loop: 72% blocked in `get_command_buffer` waiting for the
  24-entry command-buffer pool to drain (device-side per-submit cost).

### <a id="r5"></a>R5 — M1, codegen inspection, 2026-10-29
Compared RADV ISA (`RADV_DEBUG=shaders`) with HIP ISA for the same JIT'd
nbody kernels:

| Kernel | HIP | RADV |
|--------|-----|------|
| accel inner loop | 5 `v_fmac_f32`, 3 mul, 1 add; `p[j]` via `s_load_b128` | 0 FMA, 9 mul, 6 add, 3 sub; `p[j]` via `global_load_b96`×3 + `global_load_b32` |
| accel with `--cl-mad-enable` | — | 5 `v_fmac_f32`, 4 mul, 1 add, 3 sub |
| `single_task` sum loop | `s_load_b32` (SMEM) | `global_load_b32` (VMEM), 64-bit IV |
| update | 32-bit `v_mad_u64_u32` addressing, FMAs | 64-bit `v_lshlrev_b64` + add chains, no FMA |

All SPIR-V float ops carry `NoContraction` although the input LLVM IR carries
`contract` fast-math flags.

### <a id="r6"></a>R6 — M1, HeCBench suite, 2026-10-29
Built with `tools/hecbench_run.sh` instructions (`--acpp-targets=generic`),
warm-up run discarded, one measured run per device. Bold = Vulkan worse than
the native backend on the same hardware by >20%. Ratio columns are
Vulkan/native time (>1 means Vulkan slower).

Only correctness-verified pairs are included (see R7). Sizes were reduced
from HeCBench defaults for some benchmarks to keep CPU runs short.

| Benchmark (args) | Metric | HIP | RADV | RADV/HIP | OpenMP | llvmpipe | lvp/OMP |
|---|---|---|---|---|---|---|---|
| bfs (generated 1M-node graph, ~6 edges/node) | kernel time | 16.5 ms | 16.8 ms | 1.02 | 14.6 ms | 43.8 ms | **3.0** |
| floydwarshall (1024 100 16) | avg time / run (1024 launches) | 87.8 ms | **120 ms** | **1.37** | 142 ms | **800 ms** | **5.6** |
| gaussian (-q -t -s 4096) | total kernel time (~8k launches) | 4.16 s | 3.83 s | 0.92 | 15.1 s | **44.9 s** | **3.0** |
| pathfinder (100000 1000 5) | total kernel time | 2.11 s | 0.98 s | 0.47 | (excluded, R7) | 168.5 s | — |
| memcpy (8192), 64 B–64 KB | per-copy latency, H2D | 5.5–8.9 µs | **127–159 µs** | **~20** | 5.3–6.8 µs | **35–75 µs** | **~7** |
| memcpy, 64 B–64 KB | per-copy latency, D2H | 12–20 µs | **117–151 µs** | **~9** | 5.4–6.6 µs | **24–64 µs** | **~6** |
| memcpy, 1 MB | H2D / D2H | 45 / 48 µs | **2406 / 456 µs** | **53 / 9.4** | 14 / 14 µs | **422 / 390 µs** | **30 / 28** |
| memcpy, 2 MB | H2D / D2H | 72 / 76 µs | **792 / 2641 µs** | **11 / 35** | 20 / 20 µs | **1371 / 780 µs** | **68 / 39** |
| black-scholes (100) | avg kernel | 36.8 ms | 32.5 ms | 0.88 | 267 ms | **482 ms** | **1.8** |
| black-scholes (100) | "processing time" (incl. transfers) | 446 ms | **798 ms** | **1.79** | 562 ms | **2690 ms** | **4.8** |
| binomial | avg kernel | 8.10 ms | 7.35 ms | 0.91 | 83.4 ms | **385 ms** | **4.6** |
| mandelbrot (1000) | avg kernel | 1.01 ms | **1.61 ms** | **1.60** | 32.2 ms | 30.8 ms | 0.96 |
| mandelbrot (1000) | avg "parallel time" (kernel + 8 MB D2H) | 2.57 ms | **34.4 ms** | **13.4** | 32.6 ms | 41.6 ms | 1.28 |
| softmax (100000 784 1 100) | avg kernel | 15.7 ms | 11.9 ms | 0.76 | 27.4 ms | **95.4 ms** | **3.5** |
| histogram (--i=100), first config | smem atomics | 297 µs | **429 µs** | **1.44** | 897 µs | **1296 µs** | **1.44** |
| histogram (--i=100), first config | gmem atomics | 473 µs | **673 µs** | **1.42** | 888 µs | **2178 µs** | **2.45** |
| atomicReduction | GB/s (best of variants) | 85.2 | 82.1 | ~1.04 | 39.5 | 20.7 | **~1.9** |
| su3 (-i 100 -l 32 -t 128 -v 3 -w 1) | GFLOP/s | 32.3 | **12.3** | **2.62** | 4.57 | 7.82 | 0.58 |

Notes:
* RADV matches or beats HIP on compute-heavy kernels with few launches:
  pathfinder (2.1× faster), softmax, binomial, gaussian, black-scholes
  kernel. The HIP pathfinder result was not investigated.
* Transfer-dominated metrics (black-scholes processing time, mandelbrot
  parallel time) track IMPR-3.
* memcpy latencies at 1–2 MB on RADV are erratic (H2D 1 MB 2.4 ms but 2 MB
  0.8 ms), consistent with per-copy allocation cost of large staging BOs.

### <a id="r7"></a>R7 — M1, HeCBench correctness triage, 2026-10-29
Correctness of every pair in R6 was confirmed by the benchmark's own check
(bfs, floydwarshall, gaussian, binomial, mandelbrot, softmax, histogram,
atomicReduction, su3 via `assert`), by comparing printed results
(black-scholes output price at index 25000000 agrees to 6 digits on all
devices), or by building a print variant and diffing between devices
(pathfinder with `-DBENCH_PRINT`, 100000 100 5: HIP, RADV and llvmpipe
bit-identical; **OpenMP differs**, so pathfinder OpenMP data is excluded).
Benchmarks excluded entirely are listed in [BUG-5](#bug-5).

### <a id="r8"></a>R8 — M1, su3 ISA inspection, 2026-10-29
Kernel computes `size_t i = id/36; j = (id%36)/9; k = (id%9)/3; l = id%3`.

| | HIP | RADV |
|---|---|---|
| Integer div/mod | 32-bit magic-number multiply (`v_mad_u64_u32` ×5) | fully expanded 64-bit division: 125 `v_cmp_le_u64`, 334 `v_cndmask_b32`, 137 `v_add_co_u32`, 27 `s_bfm_b64` |
| FP64 complex multiply-add | 6 `v_fma_f64` | 12 `v_mul_f64` + 10 `v_add_f64` (no FMA) |
| Memory | 6 `global_load_b128` | 6 `global_load_b128` |

### <a id="r9"></a>R9 — M2, `tools/mb.cpp` (n=16000), 2026-10-29
`tools/run_tool.sh mb 16000`: wall clock, median of 3 runs after a warm-up
run (µs unless noted). Columns are grouped by hardware: MTL iGPU, Ada dGPU,
CPU.

| Metric | OpenCL MTL | Vulkan MTL | CUDA Ada | Vulkan Ada | OpenCL CPU | llvmpipe | OpenMP |
|--------|-----|------|------|------|------|------|------|
| Empty kernel, host submit cost (pipelined, 500×) | 6.8 | 32.3 | 4.0 | 17.6 | 32.5 | 27.3 | 1.1 |
| Empty kernel, end-to-end pipelined | 41.6 | 33.7 | 5.2 | 17.9 | 32.5 | 28.6 | 40.7 |
| Empty kernel + `wait()` | 72 | 81 | 10 | 37 | 43 | 39 | 46 |
| D2H 4 B memcpy + wait | 106 | 138 | 14 | 571 | 11 | 21 | 30 |
| H2D 4 B memcpy + wait | 105 | 143 | 9.5 | 576 | 10 | 23 | 32 |
| H2D 625 KB (first copy into a new allocation) | 242 | 1731 | 143 | 753 | 158 | 528 | 135 |
| nbody accel kernel + wait | 2.46 ms | 8.69 ms | 5.48 ms | 7.76 ms | 23.2 ms | 61.9 ms | 71.8 ms |
| nbody update kernel + wait | 83 | 130 | 25 | 200 | 39 | 77 | 209 |
| `single_task` serial sum of 16000 floats + wait | 1445 | 1658 | 226 | 805 | 24 | 267 | 14 |

* On the MTL iGPU the native OpenCL backend has launch/sync costs of the
  same order as Vulkan (its pipelined launches are even slower).
* OpenMP's end-to-end cost per empty kernel is 41 µs on this 22-thread
  hybrid CPU (5.7 µs on M1).
* The nbody update kernel on Vulkan Ada varies between 60 and 225 µs from
  run to run (CUDA 21–28 µs). A one-off variant with a profiling queue
  measured 37 µs of device time vs 19.5 µs on CUDA: the clspv SPIR-V copies
  the 40-byte struct with 10 scalar loads and 10 scalar stores, the CUDA PTX
  with 5 `ld.global.v2` and 4 `st.global.v2` ([IMPR-12](#impr-12)); the
  rest is submission and wake-up latency. Its cost grows ~2.3× faster with
  n than on CUDA (n=256000: 615 vs 221 µs), and ~3.1× faster on ANV than
  on OpenCL (1521 vs 484 µs).

### <a id="r10"></a>R10 — M2, `tools/bw.cpp`, 2026-10-29
µs per kernel for 50 back-to-back dependent kernels, median of 3:

| Buffer | OpenCL MTL | Vulkan MTL | CUDA Ada | Vulkan Ada | OpenCL CPU | llvmpipe | OpenMP |
|--------|-----|------|------|------|------|------|------|
| 1 MB | 98 | 156 | 12.8 | 22.1 | 36 | 108 | 56 |
| 4 MB | 194 | 201 | 34.0 | 45.3 | 85 | 318 | 89 |
| 16 MB | 565 (59 GB/s) | 559 (60 GB/s) | 301 (111 GB/s) | 303 (111 GB/s) | 333 (101 GB/s) | 1240 (27 GB/s) | 393 (85 GB/s) |
| 64 MB | 2111 (63.6 GB/s) | 2120 (63.3 GB/s) | 1203 (111.5 GB/s) | 1189 (112.9 GB/s) | 2858 (47 GB/s) | 4339 (31 GB/s) | 3385 (40 GB/s) |
| 256 MB | 8344 (64.3 GB/s) | 8485 (63.3 GB/s) | 4845 (110.8 GB/s) | 4744 (113.2 GB/s) | 11689 (46 GB/s) | 19889 (27 GB/s) | 12906 (42 GB/s) |

64 MB with 4/6/8/12/16/22 threads (`LP_NUM_THREADS`, `OMP_NUM_THREADS`,
`CL_CONFIG_CPU_TBB_NUM_WORKERS`): llvmpipe 20/17/18/21/25/29 GB/s, OpenMP
47/49/45/35/44/44 GB/s, OpenCL CPU 32/45/50/53/53/52 GB/s.

### <a id="r11"></a>R11 — M2, host↔device copies, 2026-10-29/30
`tools/xfer.cpp` (memcpy + `wait()`, pageable host memory, 2 runs; bold =
D2H through uncached staging memory, [IMPR-13](#impr-13)):

| Size | OpenCL MTL | Vulkan MTL | CUDA Ada | Vulkan Ada | OpenCL CPU | llvmpipe | OpenMP |
|------|-----|------|------|------|------|------|------|
| 16 B – 4 KB, H2D / D2H (µs) | 99–107 / 102–105 | 179–208 / 182–235 | 9.4–12 / 14–15 | 565–738 / 570–592 | 9–24 / 10–22 | 22–25 / 20–22 | 29–38 / 29–42 |
| 1 MB, H2D / D2H (µs) | 220 / 235 | 580 / **4643** | 170 / 202 | 1018 / 762 | 40 / 39 | 271 / 278 | 34 / 35 |
| 16 MB, H2D / D2H (GB/s) | 11.1 / 11.0 | 3.3 / **0.16** | 8.2 / 8.1 | 2.6 / 3.0 | 67 / 66 | 3.9 / 4.0 | 44 / 49 |
| 256 MB, H2D / D2H (GB/s) | 13.1 / 13.0 | 3.6 / **0.16** | 8.4 / 8.3 | 2.3 / 2.1 | 26 / 25 | 4.2 / 4.2 | 21 / 21 |

`tools/memtype.cpp 64` (host `memcpy` through `vkMapMemory`, no GPU work):

| Device | Memory type (AdaptiveCpp use) | Host write | Host read | allocate+map+unmap+free, 4 KB |
|--------|------|------|------|------|
| ANV | `DL\|HV\|HC`, uncached (staging: preferred `HV\|HC\|HCACHED` doesn't exist) | 20.3 GB/s | **0.16 GB/s** | 10 µs |
| ANV | `DL\|HV\|HCACHED`, not coherent (unused) | 14.8 GB/s | 15.2 GB/s | 5–8 µs |
| NVIDIA | `DL\|HV\|HC`, ReBAR VRAM (device USM) | 11.4 GB/s | **0.02 GB/s** | 119 µs |
| NVIDIA | `HV\|HC`, system | 20.1 GB/s | 0.17 GB/s | 433 µs |
| NVIDIA | `HV\|HC\|HCACHED`, system (staging) | 20.1 GB/s | 21.6 GB/s | **334 µs** |
| llvmpipe | all flags | 20.2 GB/s | 22.1 GB/s | 1.6 µs |

`strace -f -c tools/lat` (net of the K=M=0 baseline):
* Vulkan MTL: ~3.3 ioctl per launch; 11 ioctl + 3 futex + mmap + munmap per
  4-byte D2H copy.
* Vulkan NVIDIA: no syscalls per launch (user-mode submission); per copy
  10 ioctl + 3 futex + 2 poll + `openat` + `close` + mmap + munmap.
* llvmpipe: ~79 futex per launch; 19 futex + mmap + munmap per copy.

gdb sampling (`samp.sh`, 150 samples) of `lat 0 <many>` on NVIDIA: the
submitting thread spends 79% in `vk_node_event::wait` and 16% in
`vk_allocator::staging_allocate` (`vkAllocateMemory`); the host worker
thread spends ~40% of its time in `cleanup_staging_buffers` →
`vk_staging_allocation_map::erase` (`vkFreeMemory`).

### <a id="r12"></a>R12 — M2, `tools/idx.cpp` (16M items, 64 iterations) and IR, 2026-10-29/30
Time per kernel (ms), 2 runs:

| | OpenCL MTL | Vulkan MTL | CUDA Ada | Vulkan Ada | OpenCL CPU | llvmpipe | OpenMP |
|---|---|---|---|---|---|---|---|
| `size_t` index maths | 19.8 | 49.8 | 17.5 | **246.9** | 199 | 291 | 205–239 |
| `unsigned` index maths | 12.2 | 12.9 | 17.5 | 16.4 | 170 | 168 | 101–111 |
| ratio | 1.62 | 3.85 | 1.00 | **15.0** | 1.17 | 1.73 | 2.0–2.2 |

* clspv SPIR-V of the `size_t` kernel contains `OpUDiv %ulong %x %ulong_36`,
  `OpUMod %ulong %x %ulong_9` and `OpUMod %ulong %x %ulong_3`: division by a
  constant is left to the Vulkan driver's compiler.
* The CUDA backend emits identical 32-bit PTX (`mul.hi.u32`) for both
  variants. `ACPP_S2_DUMP_IR_FINAL=1` shows why: in the PTX flavour the
  global id is built from i32 `ctaid`/`tid` plus the
  `GlobalSizesFitInI32OptPass` range `llvm.assume`s, and InstCombine narrows
  the 64-bit `add`/`udiv` to i32 (`%narrow.i = add nuw i32 ...`,
  `udiv i32 ..., 36`). In the clspv flavour `get_group_id`/`get_local_id`
  are opaque i64 calls (`src/libkernel/sscp/clspv/core.cpp`): the same range
  comparisons are present but there is no `zext` from i32 to narrow, so
  `udiv i64` survives (the dead `assume`s are dropped later by
  `RemoveUnusedIntrinsicsPass`).

### <a id="r13"></a>R13 — M2, `tools/prof.cpp`, BUG-2 and BUG-3 checks, 2026-10-29/30
`prof 64` (in-order profiling queue, every event waited on; µs, 2 runs):

| Device | Wall clock | Event profiling | Wall / profiled |
|--------|------|------|------|
| Vulkan MTL (`timestampPeriod` 52.08) | 3285–3425 | 74.7–78.5 | **44** |
| Vulkan Ada (period 1) | 1927–1957 | 1635–1637 | 1.2 |
| CUDA Ada | 1142 | 1132 | 1.01 |
| llvmpipe (period 1) | 20074–20236 | 18823–19080 | 1.06 |
| OpenMP | 12465–12587 | 12200–12345 | 1.02 |
| OpenCL MTL / CPU | abort: `Operation not profiled: No execution end timestamp available` | | |

The kernel is a 64-step `x = x*1.0001f + 0.5f` chain: Vulkan Ada takes
1637 µs vs 1132 µs on CUDA ([IMPR-5](#impr-5)).

* BUG-2 (`tools/bug2.cpp`, 2000 fire-and-forget kernels on a profiling
  queue): the validation layer reports
  `VUID-vkDestroyQueryPool-queryPool-00793` on ANV, NVIDIA and llvmpipe.
  ANV and NVIDIA complete; llvmpipe segfaults with the layer and, without
  it, aborts or hangs (2 of 3 runs hit a 300 s timeout).
* BUG-3: `omp;vulkan` and `omp;vulkan:llvmpipe` select the Vulkan backend /
  device. `omp;vulkn` prints `'vulkn' is not a known backend name` and
  `Could not parse value of environment variable: ACPP_VISIBILITY_MASK`,
  then loads *all* backends. `omp;vk:NoSuchDevice` loads Vulkan with no
  devices and no message.

### <a id="r14"></a>R14 — M2, HeCBench suite, 2026-10-29/30
`tools/hecbench_suite.sh` (R6's arguments; bfs on two generated graphs),
warm-up run discarded, one measured run per device; the GPU devices were
run a second time (pass 2). Ratios are Vulkan/native time (>1 means Vulkan
slower; inverted for GB/s and GFLOP/s). Bold = Vulkan >20% slower than the
native backend on the same hardware. Only correctness-verified pairs
(R15) have numbers.

| Benchmark (args) | Metric | OpenCL MTL | Vulkan MTL | VK/OCL | CUDA Ada | Vulkan Ada | VK/CUDA | OpenCL CPU | llvmpipe | lvp/OCL | OpenMP | lvp/OMP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bfs (random graph, 1M nodes, ~6 edges/node) | kernel time | 10.6 ms | 13.5 ms | **1.27** | 3.50 ms | 6.63 ms | **1.89** | 99.4 ms | 30.0 ms | 0.30 | 17.1 ms | **1.75** |
| bfs (1000×1000 grid, 1999 levels) | kernel time | 415 ms | 1121 ms | **2.70** | 145 ms | 297 ms | **2.05** | 459 ms | 1346 ms | **2.93** | 777 ms | **1.73** |
| bfs (1000×1000 grid) | process wall time | 2.31 s | 3.41 s | **1.48** | 1.71 s | 4.11 s | **2.41** | 1.91 s | 2.81 s | **1.47** | 2.21 s | **1.27** |
| floydwarshall (1024 100 16) | avg time / run | 106 ms | 102 ms | 0.96 | 38.5 ms | 48.8 ms | **1.27** | 145 ms | 660 ms | **4.54** | 145 ms | **4.55** |
| gaussian (-q -t -s 4096) | total kernel time | 3.92 s | 4.00 s | 1.02 | 2.27 s | 2.32 s | 1.02 | 2.62 s | 28.8 s | **11.0** | 5.85 s | **4.92** |
| pathfinder (100000 1000 5) | total kernel time | 1.22 s | 3.35 s | **2.73** | 2.08 s | 3.46 s | **1.66** | 232 s | 103 s | 0.44 | 113 s | 0.91 |
| memcpy (8192), pipelined | per copy, 64 B, H2D / D2H | 30 / 31 µs | 181 / 186 µs | **6.0 / 6.0** | 4.2 / 9.5 µs | 491 / 542 µs | **117 / 57** | 5.5 / 6.5 µs | 16 / 16 µs | **2.9 / 2.5** | 29 / 29 µs | 0.54 / 0.56 |
| memcpy | per copy, 64 KB, H2D / D2H | 38 / 38 µs | 298 / 594 µs | **7.8 / 16** | 12 / 21 µs | 580 / 587 µs | **49 / 28** | 23 / 23 µs | 32 / 35 µs | **1.35 / 1.49** | 29 / 29 µs | 1.09 / 1.19 |
| memcpy | per copy, 1 MB, H2D / D2H | 83 / 105 µs | 579 / 5305 µs | **7.0 / 50** | 123 / 186 µs | 1501 / 1327 µs | **12 / 7.1** | 37 / 38 µs | 331 / 335 µs | **8.9 / 8.9** | 32 / 32 µs | **10 / 10** |
| memcpy | per copy, 2 MB, H2D / D2H | 94 / 106 µs | 868 / 9615 µs | **9.2 / 91** | 242 / 310 µs | 2589 / 2271 µs | **11 / 7.3** | 52 / 52 µs | 689 / 728 µs | **13 / 14** | 41 / 41 µs | **17 / 18** |
| black-scholes (100) | avg kernel | 26.4 ms | 26.3 ms | 1.00 | 16.8 ms | 19.4 ms | 1.15 | 85.6 ms | 276 ms | **3.22** | 114 ms | **2.43** |
| black-scholes (100) | "processing time" (incl. transfers) | 591 ms | 2199 ms | **3.72** | 503 ms | 1024 ms | **2.04** | 380 ms | 1746 ms | **4.60** | 293 ms | **5.95** |
| binomial | avg kernel | 4.01 ms | 7.54 ms | **1.88** | 5.23 ms | 9.93 ms | **1.90** | 432 ms¹ | 699 ms¹ | **1.62** | 515 ms¹ | **1.36** |
| mandelbrot (1000) | avg kernel | 0.714 ms | 2.48 ms | **3.48** | 0.750 ms | 2.75 ms | **3.66** | 4.28 ms | 28.1 ms | **6.56** | 23.5 ms | 1.20 |
| mandelbrot (1000) | avg "parallel time" (kernel + 8 MB D2H) | 2.45 ms | 45.0 ms | **18.4** | 3.23 ms | 13.7 ms | **4.24** | 5.87 ms | 36.0 ms | **6.14** | 24.7 ms | **1.46** |
| softmax (100000 784 1 100) | avg kernel | crash² | FAIL³ | — | 5.52 ms | 5.47 ms | 0.99 | crash² | 88.3 ms | — | 19.1 ms | **4.62** |
| histogram (--i=100), first config | smem atomics | 696 µs | 1003 µs | **1.44** | 72.8 µs | 134 µs | **1.84** | 293 µs | 1889 µs | **6.45** | 1583 µs | 1.19 |
| histogram (--i=100), first config | gmem atomics | 412 µs | 1363 µs | **3.31** | 338 µs | 376 µs | 1.11 | 1057 µs | 2259 µs | **2.14** | 1512 µs | **1.49** |
| atomicReduction | GB/s (best of variants) | 78.1 | 77.6 | 1.01 | 121 | 123 | 0.99 | 41.6 | 29.7 | **1.40** | 42.8 | **1.44** |
| su3 (-i 100 -l 32 -t 128 -v 3 -w 1) | GFLOP/s | 49.6 | 48.6 | 1.02 | 30.8 | 22.9 | **1.34** | 26.5 | 11.9 | **2.22** | 6.33 | 0.53 |
| nbody (16000 30) | GFLOP/s | 1339 | 539 | **2.48** | 936 | 649 | **1.44** | 278⁴ | 86.6 | **3.21** | 81.3 | 0.94 |

¹ 20-launch build of binomial (`NUM_ITERATIONS 20`); the default 1000
launches take ~450–500 s on the OpenCL CPU and OpenMP and exceed 600 s on
llvmpipe. GPU columns use the default build.
² OpenCL backend SIGFPE before any output.
³ Wrong results at this commit (ANV sub-group size, R15).
⁴ Reports `FAIL` because of the energy tolerance artefact described in R1.

Notes:
* Pass 2 reproduced the GPU ratios within 3% except bfs on MTL (random
  1.41, grid 2.33), histogram on MTL (smem 1.10, gmem 2.21: Vulkan MTL
  varies 713–1003 / 910–1363 µs between runs), black-scholes processing
  time on MTL (4.16) and mandelbrot kernel on Ada (4.12).
* Parity on both GPUs where kernels are long and simple: floydwarshall,
  gaussian, black-scholes kernel, atomicReduction and su3 on MTL;
  gaussian, softmax, atomicReduction and black-scholes kernel (1.15) on
  Ada; and large-buffer bandwidth on both (R10).
* pathfinder and binomial are dominated by barrier semantics on NVIDIA
  ([IMPR-14](#impr-14)); nbody by math lowering and contraction on NVIDIA
  ([IMPR-11](#impr-11), [IMPR-5](#impr-5)) and by addressing on ANV
  ([FUND-2](#fund-2), [IMPR-6](#impr-6)); see R16.
* mandelbrot's avg kernel combines ANV's first-use cost of the buffer
  allocated afresh in every iteration ([IMPR-15](#impr-15)), a 1.2–1.8×
  slower loop on both GPUs (R16) and possibly lower GPU clocks after the
  slow staged copies between kernels.
* Transfer-dominated metrics follow [IMPR-3](#impr-3) and, for D2H on ANV,
  [IMPR-13](#impr-13) (black-scholes processing time, mandelbrot parallel
  time, memcpy D2H at 1–2 MB).
* The grid bfs has 1999 levels, each with a 1-byte H2D copy, two kernels
  and a 1-byte D2H copy: Vulkan's per-operation costs show up as 2.0–2.7×
  in kernel time and 1.5–2.4× in process wall time.
* RADV was 2.1× faster than HIP on pathfinder on M1, but both Vulkan
  drivers are slower than native on M2.

### <a id="r15"></a>R15 — M2, HeCBench correctness triage, 2026-10-30
* Self-checks pass on every device that ran for bfs (both graphs),
  floydwarshall, gaussian, binomial, mandelbrot, histogram and
  atomicReduction; su3's `assert`s hold everywhere; nbody reports the R1
  tolerance artefact on the OpenCL CPU only.
* black-scholes: output price at index 25000000 agrees to 6 digits
  (10.84791x) on all 7 devices.
* pathfinder `-DBENCH_PRINT` (100000 100 5): bit-identical on all 7
  devices, including OpenMP (which differed on M1).
* lavaMD `-DOUTPUT` (-boxes1d 10), max relative difference vs CUDA:
  OpenCL MTL 7e-6, OpenCL CPU 1.2e-5, OpenMP 1.3e-5; ANV **42**, NVIDIA
  **21**, llvmpipe **9.9**.
* lud (-s 1024 -v): OpenCL MTL verifies; ANV, NVIDIA and llvmpipe report
  ~32k mismatches starting at (1, 16); OpenMP 2052 from (1, 0); CUDA hangs;
  OpenCL CPU segfaults.
* Sub-group size on ANV: re-running with the Vulkan backend built from
  `vk/per-kernel-subgroup-size` (`055fe70d`, which requires the reported
  sub-group size through `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`;
  loaded via `LD_LIBRARY_PATH` from a copy of the install's `lib/`): softmax
  and layernorm pass; atomicCost, shuffle and lud are unchanged.
* atomicCost FP64 sub-test: the backend prints
  `vk_executable_object: unexpected SPIRV capability 6034` and runs the
  kernel anyway ([BUG-6](#bug-6)).
* scan: clspv aborts (see BUG-5) on all three Vulkan drivers and every
  further launch re-runs the failing JIT, so the process looks hung
  ([BUG-7](#bug-7)).
* Not Vulkan, recorded to explain missing OpenCL data: softmax SIGFPE on
  both OpenCL devices; histogram/atomicCost/babelstream segfault after the
  runtime is re-initialised within the process; lud segfaults on the OpenCL
  CPU; CUDA hangs in lud.

### <a id="r16"></a>R16 — M2, codegen experiments, 2026-10-30
1. nbody acceleration kernel ISA (`INTEL_DEBUG=cs`, IGC dump):

   | | OpenCL (IGC) | Vulkan (ANV) |
   |---|---|---|
   | SIMD width | 32 | 16 |
   | `p[j]` (uniform address) | one single-lane `load.ugm.d32x3t.a32` with a 32-bit stateful (BTI) offset | per-lane `send(16) ... load, a64, d32, V3` + `V1` gathers |
   | Loop counter / address | 32-bit, scalar `j*40` | `int j` widened to i64 (SPIR-V `OpPhi %ulong`); `p + j*40` recomputed with emulated 64-bit `mul`/`mach`/carry |
   | FP | `mad` | separate `mul`/`add` |

   ANV's inner loop has ~45 integer instructions for ~20 float ones.
2. `tools/rsq.cpp` (ms, steady state):

   | | OpenCL MTL | Vulkan MTL | CUDA Ada | Vulkan Ada | OpenCL CPU | llvmpipe | OpenMP |
   |---|---|---|---|---|---|---|---|
   | `sycl::rsqrt` | 3.0 | 9.3–10.1 | 5.42 | 7.58 | 21.9 | 68 | 62–66 |
   | `sycl::native::rsqrt` | 2.77 | 7.9–8.5 | 5.42 | 6.54 | 21.9 | 54–59 | 68–72 |
   | `1 / sycl::sqrt` | 2.26 | 9.1–9.2 | 5.42 | 7.57 | 17.5 | 67 | 69–73 |

   clspv SPIR-V for `sycl::rsqrt` is `Sqrt`, range scaling, `OpFDiv 1.0, x`
   and selects; every fp32 division carries the same scaling.
3. Extra clspv flags through `tools/clspv_shim.c` (`sycl::rsqrt` variant,
   fresh JIT cache, 3 runs): NVIDIA 7.72 ms → 6.55 ms (`--cl-mad-enable`)
   → 5.95 ms (`--cl-native-math`; CUDA 5.42); ANV ~10 → ~10 → ~7.7 ms
   (OpenCL ~3); llvmpipe ~70 → ~70 → ~61 ms.
4. Barrier semantics A/B (`tools/patch_barrier.sh`: `0x148` → `0x108` in the
   cached SPIR-V, i.e. without `UniformMemory`), alternating runs:

   | | Vulkan MTL | Vulkan Ada | Native |
   |---|---|---|---|
   | pathfinder (100000 1000 5), total kernel | 3.38 → 3.03 s | 3.45 → 2.18 s | OpenCL 1.22 s, CUDA 2.08 s |
   | binomial, avg kernel | 7.67 → 6.80 ms | 10.09 → 7.06 ms | OpenCL 4.01 ms, CUDA 5.23 ms |
   | histogram (first config, smem and gmem) | unchanged | unchanged | |

   ANV emits a UGM (global) fence before each barrier in addition to the
   SLM fence; IGC emits only `fence.slm.none.group`.
5. `tools/mandel.cpp` (kernel + `wait()`, steady state): reused buffer
   OpenCL MTL 0.70–0.73, Vulkan MTL 0.85–1.05, CUDA 0.71, Vulkan Ada
   1.15–1.28 ms; fresh 8 MB buffer every launch: OpenCL MTL 0.67–0.69,
   **Vulkan MTL 1.85–2.08**, CUDA 0.72, Vulkan Ada 1.10–1.15 ms.
   `--cl-mad-enable` gives −10% on NVIDIA and nothing on ANV. The rest of the
   NVIDIA gap (~1.5×) is unexplained: the CUDA loop is not unrolled and
   differs from the SPIR-V mainly by its two FMAs.
6. `single_task` serial sum (`mb.cpp`): the CUDA PTX is runtime-unrolled ×8
   with `ld.global.v4.b32`/`v2` loads; the clspv SPIR-V is the plain loop
   with one scalar `OpLoad` per iteration and a 64-bit induction variable.
7. llvmpipe (`GALLIVM_DEBUG=asm`, `bw.cpp`): the load of `a[i]` is two
   `vpgatherqd` (64-bit addresses), the store of `b[i]` is 8 scalar stores
   via `vextractps`/`vpextrq`/`vmovss`.

### <a id="r17"></a>R17 — M2, minimal reproducers for the bug reports, 2026-10-30
Each report in [`../bug-reports/`](../bug-reports/) contains its reproducer and
the verbatim output; highlights:
* BUG-2: with every event waited on, the validation layer reports nothing
  (NVIDIA, llvmpipe); dropping the events reproduces VUID-00793.
* BUG-6: `atomic_ref<double>::fetch_add` on 1024 work-items: ANV `-nan`,
  llvmpipe aborts in its LLVM back end, NVIDIA 1024. The Vulkan backend
  reports `aspect::fp64 == aspect::atomic64 == false` on all three devices.
* BUG-7/BUG-11: a 40-line scan up-sweep with a `local_accessor<char>` makes
  clspv abort on all three Vulkan drivers; each of 3 launches re-runs the JIT
  (~1.74 s each) and the kernel never runs; OpenCL and CUDA compile once and
  give the right result. `llvm-reduce` on the IR dumped from scan-sycl gives an
  8-line module that aborts in `clspv-opt --passes=simplify-pointer-bitcast`:
  an `i8` GEP into an `[N x i32]` global at a byte offset that is not a
  multiple of 4 (1, 2, 3, 5, −1, −2, −3 abort; ±4 do not). Backtrace:
  `SimplifyPointerBitcastPass::runOnImplicitGEP` →
  `BitcastUtils::GetIdxsForTyFromOffset`.
* BUG-10: a sub-group reduction kernel on ANV is compiled SIMD16 while 32 is
  reported (`INTEL_DEBUG=cs`); 16 of 256 work-items see
  `sub_group::get_local_range() == -224`. With `vk/per-kernel-subgroup-size`
  the kernel is SIMD32 and correct.
* BUG-12/13/14 (OpenCL backend): see the reports.

### <a id="r18"></a>R18 — M2, user-level workarounds and `sycl::native` functions, 2026-10-30
Run to check the advice in
[`../fundamental-limitations/`](../fundamental-limitations/) before giving it.
`tools/uguide.cpp`: wall clock, median of 15 launches after a warm-up, second
of two runs with a fresh private JIT cache, `performance` power profile (ms):

| | OpenCL MTL | Vulkan MTL | CUDA Ada | Vulkan Ada | OpenCL CPU | llvmpipe |
|---|---|---|---|---|---|---|
| 1024-step chain `x = x*a + b` (4M work-items) | 4.80 | 8.51 | 2.77 | 6.83 | 35.2 | 533 |
| same with `x = sycl::fma(x, a, b)` | 4.43 | **351** | 2.76 | **499** | 36.4 | **5333** |
| 128-step chain `x = a / (x + b)` | 4.01 | 3.49 | 5.13 | 4.13 | 16.1 | 102 |
| same with `x = a * sycl::native::recip(x + b)` | 2.13 | 2.33 | 4.03 | 2.65 | 18.4 | 98 |
| `single_task` serial sum of 65536 floats | 4.72 | 6.41 | 0.91 | 3.15 | 0.027 | 1.05 |
| same with `#pragma unroll 8` | 1.98 | 2.56 | **2.43** | 1.14 | 0.038 | 0.28 |
| same, unrolled ×8 by hand (same summation order) | 1.83 | 2.77 | 2.43 | 1.14 | 0.094 | 0.22 |

* Between the two runs the GPU results agree within 16%, except the two
  unrolled serial sums on Vulkan Ada (2.31 / 2.88 ms in the first run, with
  an empty JIT cache). CPU results vary more (llvmpipe: up to 1.8× for the
  hand-unrolled sum, 1.3× for the division chain).
* The same binary built with `-ffast-math`: no change on the GPUs (`x*a+b`
  NVIDIA 6.80 ms, ANV 8.67 ms; `sycl::fma` unchanged).
* SPIR-V of one `sycl::fma` in a `single_task` (`spirv-dis` of the JIT
  cache): 629 lines with ~200 integer instructions; `GLSL.std.450 Fma` is only
  reached when an input is NaN or infinite, finite inputs take a software
  path. `sycl::mad` gives `OpFMul` + `OpFAdd`, both `NoContraction`, i.e. the
  same code as `a*b + c`.
* `VK_KHR_shader_fma`: NVIDIA exposes it (`shaderFmaFloat16/32/64 = true`),
  ANV and llvmpipe (Mesa 26.0.8) do not. clspv has `--spv-khr-fma=16,32,64`
  but only lowers `fma` to `OpFmaKHR` when every enabled float width is
  listed (`lib/Compiler.cpp`; fp16 and fp64 are enabled by default). Through
  `clspv_shim.c` on NVIDIA: `--spv-khr-fma=32` changes only the NaN/Inf path
  (chain still 499.5 ms); `--spv-khr-fma=16,32,64` gives a single `OpFmaKHR`
  and the chain takes **4.49 ms** (`x*a+b` 6.80 ms, CUDA 2.77 ms). The device
  feature was not enabled for this experiment; the backend printed
  `unexpected SPIRV capability 6030` (BUG-6 behaviour) and ran the kernel.
* `sycl::native` functions, one kernel each, on all three Vulkan drivers:
  `native::divide` makes the kernel fail to load (`failed to parse spirv`;
  [BUG-15](#bug-15)); `recip`, `rsqrt`, `sqrt`, `powr`, `exp`, `log` and `cos`
  give the expected values.
* `tools/jit_unroll.cpp`: `#pragma unroll 8` applied through
  `AdaptiveCpp_jit::compile_if_else(reflect<reflection_query::runtime_backend>()
  == (int)sycl::backend::vk, ...)` reaches only the Vulkan devices; the
  others run the plain loop. Two runs: Vulkan Ada 1.01 / 1.10 ms (unrolled),
  CUDA 0.89 ms (plain), llvmpipe 0.26–0.28 ms (unrolled), OpenCL CPU
  0.03 ms (plain); the MTL iGPU varied too much between runs to compare
  (Vulkan 3.3–5.3 ms unrolled, OpenCL 4.8–4.9 ms plain).

---

## Findings

Finding format: **Status**, **Summary**, **Mechanism** (code pointers),
**Impact**, **User guidance** (fundamental only) or **Proposed fix**
(improvable/bugs), **Evidence** table.

### A. Fundamental limitations (communicate to users)

Things that are inherent to Vulkan, its SPIR-V environment, or the drivers,
and which AdaptiveCpp can mitigate but not eliminate.

#### <a id="fund-1"></a>FUND-1: Per-submission cost and kernel-resolved dependencies
* **Status:** confirmed (M1 RADV, M2 NVIDIA); on the M2 iGPU the native
  OpenCL backend pays a similar cost, so it is not Vulkan-specific there
* **Summary:** Each Vulkan queue submission costs tens of µs host-side and
  dependencies between submissions are resolved by the kernel driver's
  scheduler via timeline semaphores. There is no equivalent of HIP's AQL
  barrier bit / CUDA stream ordering in the hardware queue. Batching
  (IMPR-1) amortises but does not remove this.
* **Impact:** Workloads with many small kernels or many host syncs.
* **User guidance:** Prefer fewer, larger kernels; fuse kernels; avoid
  `wait()` inside loops; avoid kernels shorter than ~100 µs.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R2, R3, R4 | M1 | confirms | RADV +40-50 µs per dependent kernel vs HIP at all sizes up to 8 MB; 3 ioctl/launch |
| R6 | M1 | confirms | floydwarshall (1024 dependent launches/run): RADV 1.37× HIP ≈ +31 µs/launch |
| R6 | M1 | partial | gaussian (~8k launches) RADV 0.92× HIP: once kernels are big enough the per-launch cost is hidden |
| R9, R10, R11 | M2 | confirms | NVIDIA: 17.9 vs 5.2 µs per pipelined launch, 37 vs 10 µs with `wait()`, +9–11 µs per dependent kernel at 1–4 MB. NVIDIA submits without syscalls, so this cost is user-space (AdaptiveCpp + driver) |
| R14 | M2 | confirms | floydwarshall NVIDIA 1.27× CUDA (+10 µs/launch); gaussian 1.02× |
| R9, R10, R14 | M2 | partial | ANV vs OpenCL on the same iGPU: 34 vs 42 µs pipelined, 81 vs 72 µs with `wait()`, 3.3 ioctl/launch; floydwarshall 0.96×, gaussian 1.02× — the native Intel stack is no faster |

#### <a id="fund-2"></a>FUND-2: No scalar (SMEM) loads for wave-uniform pointer loads on RADV and ANV
* **Status:** confirmed (M1 RADV, M2 ANV)
* **Summary:** HIP's backend proves uniform loads are unclobbered and uses
  the scalar cache. Through SPIR-V `PhysicalStorageBuffer` pointers the
  Vulkan driver compiler has no alias/read-only information and emits vector
  memory loads. ANV likewise issues a 16-lane A64 gather for a uniform
  address where IGC (OpenCL) issues one single-lane load (R16).
* **Impact:** Serial `single_task` loops (1.68 vs 1.17 ms) and uniform
  reads such as `p[j]` in nbody. Largest remaining nbody gap after IMPR-5.
  On ANV, together with the 64-bit addressing of [IMPR-6](#impr-6), it
  makes nbody 2.5× slower than OpenCL.
* **User guidance:** Avoid serial `single_task` reductions; use
  `sycl::reduction` or group algorithms.
* **Note:** partially addressable via IMPR-7; may require driver work. On
  M2 the NVIDIA `single_task` gap is explained by [IMPR-12](#impr-12) (no
  loop unrolling) instead; it is worth checking whether the HIP loop on M1
  was also unrolled. Related on Intel: physical storage buffer addresses are
  64-bit and Xe-LPG has no 64-bit integer ALU, so address arithmetic is
  emulated (R16); [IMPR-6](#impr-6) can narrow the index arithmetic, but the
  addresses themselves stay 64-bit.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R2, R5 | M1 | confirms | |
| R16 | M2 | confirms | ANV nbody accel: `p[j]` via `send(16) ... load, a64, d32, V3` per lane; IGC: one `load.ugm.d32x3t.a32 (1\|M0)` through a 32-bit stateful offset |
| R9, R16 | M2 | partial | NVIDIA `single_task` 805 vs 226 µs, but the CUDA loop is unrolled ×8 with vector loads ([IMPR-12](#impr-12)); scalar loads not shown to matter |

#### <a id="fund-3"></a>FUND-3: llvmpipe global-memory bandwidth roughly half of OpenMP
* **Status:** confirmed (M1, M2); mechanism verified on M2
* **Summary:** Streaming kernels on llvmpipe reach ~16 GB/s vs ~32 GB/s for
  OpenMP; extra threads help little (M1: no gain beyond 8 threads; M2:
  20 → 29 GB/s from 4 to 22 threads, while OpenMP saturates at 4). Lavapipe lowers
  physical-storage-buffer accesses to per-lane gathers/scatters instead of
  contiguous vector loads: in `bw.cpp` the load of `a[i]` becomes two AVX2
  `vpgatherqd` with 64-bit addresses and the store of `b[i]` becomes 8 scalar
  stores (`vextractps`/`vpextrq`/`vmovss`) (R16).
* **Impact:** Bandwidth-bound kernels on llvmpipe. Compute-bound kernels
  are fine (nbody accel is 2.5× faster than OpenMP). On M2 nbody accel is
  only 1.16× faster than OpenMP and 2.7× slower than the OpenCL CPU runtime.
* **User guidance:** llvmpipe is suited to compute-bound kernels.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R3 | M1 | confirms | 64-256 MB: 16-17 GB/s vs 27-32 GB/s |
| R6 | M1 | partial | atomicReduction (bandwidth-bound) 20.7 vs 39.5 GB/s best case; but llvmpipe is also slow on non-bandwidth-bound kernels, see FUND-5 |
| R10 | M2 | confirms | 64–256 MB: 27–31 GB/s vs OpenCL CPU 46–47 and OpenMP 40–42 GB/s; with 4 threads 20 GB/s vs 47 GB/s (OpenMP): ~2.3× less per thread |
| R14 | M2 | confirms | atomicReduction 29.7 vs 41.6 (OpenCL CPU) / 42.8 (OpenMP) GB/s |
| R16 | M2 | confirms | mechanism: `GALLIVM_DEBUG=asm` shows gathers/scatters for PSB accesses |

#### <a id="fund-4"></a>FUND-4: No host / shared USM
* **Status:** open (design)
* **Summary:** `malloc_host` / `malloc_shared` throw because mapped
  pointers can't be used as physical pointers in kernels
  (`vk_allocator::raw_allocate_optimized_host` / `raw_allocate_usm`).
  Every copy involving a host pointer goes through staging.
* **User guidance:** Use device USM with large, infrequent explicit copies.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| code inspection | — | confirms | |
| R11 | M2 | confirms | consequence measured: every host copy is staged ([IMPR-3](#impr-3), [IMPR-13](#impr-13)); no pinned path to fall back on |

#### <a id="fund-5"></a>FUND-5: llvmpipe slower than OpenMP on most real kernels
* **Status:** confirmed (M1, M2); mechanism only partly known (FUND-3)
* **Summary:** Although llvmpipe beats OpenMP on nbody (compute-bound,
  regular), it is 1.8–5.6× slower on most HeCBench kernels, including ones
  with no barriers, local memory or atomics (floydwarshall, gaussian, bfs,
  black-scholes). Exceptions: mandelbrot (parity), su3 (faster, as OpenMP
  is slow there). Against the Intel OpenCL CPU runtime (M2) llvmpipe is
  1.4–11× slower on most kernels; the exceptions (bfs on a random graph,
  pathfinder) are kernels where that runtime is itself unusually slow.
* **Impact:** llvmpipe is not a general substitute for the OpenMP backend.
* **User guidance:** Use the OpenMP backend for CPU execution;
  llvmpipe is mostly useful for testing Vulkan code paths without a GPU.
* **To investigate:** split per-launch cost (FUND-1/IMPR-1, ~35 µs) from
  per-work-item cost; check lavapipe's handling of PSB loads (FUND-3),
  64-bit index maths (IMPR-6) and workgroup scheduling for 2D ranges.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R6 | M1 | confirms | lvp/OMP: bfs 3.0, floydwarshall 5.6, gaussian 3.0, black-scholes 1.8, binomial 4.6, softmax 3.5, histogram 1.4–2.5; mandelbrot 0.96, su3 0.58 |
| R14 | M2 | confirms | lvp/OMP: bfs 1.7, floydwarshall 4.6, gaussian 4.9, black-scholes 2.4, binomial 1.4, softmax 4.6, histogram 1.2–1.5, mandelbrot 1.2; pathfinder 0.91, nbody 0.94, su3 0.53 |
| R14 | M2 | confirms | lvp/OpenCL CPU: gaussian 11, mandelbrot 6.6, histogram 2.1–6.5, floydwarshall 4.5, nbody 3.2, black-scholes 3.2, bfs (grid) 2.9, su3 2.2, binomial 1.6, atomicReduction 1.4; bfs (random) 0.30 and pathfinder 0.44 |

#### <a id="fund-6"></a>FUND-6: Correctly rounded `fma` needs a software routine without `VK_KHR_shader_fma`
* **Status:** open (M2)
* **Summary:** SYCL 2020 requires `sycl::fma` to be correctly rounded. The
  Vulkan spec only guarantees `GLSL.std.450 Fma` the precision of "OpFMul
  followed by OpFAdd"; `OpFmaKHR` from `VK_KHR_shader_fma` is "Correctly
  rounded". Devices without the extension (ANV and llvmpipe with Mesa
  26.0.8) need a software routine. AdaptiveCpp currently uses the routine on
  every device, including NVIDIA, which has the extension
  ([IMPR-16](#impr-16)).
* **Impact:** 1024-step dependent chain: ANV 351 ms with `sycl::fma` vs
  8.5 ms with `x*a+b`; llvmpipe 5.3 s vs 0.53 s.
* **User guidance:** Write `a*b + c` instead of `sycl::fma` unless the
  correctly rounded result is needed.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R18 | M2 | confirms | `uguide.cpp`; `vulkaninfo` (`VK_KHR_shader_fma` on NVIDIA only); Vulkan spec precision table |

### B. Improvable performance (actionable in AdaptiveCpp)

Ordered roughly by expected impact on M1; IMPR-11 onwards were added from
M2 and are not ranked.

#### <a id="impr-1"></a>IMPR-1: One command buffer + one vkQueueSubmit per operation
* **Status:** confirmed (M1, M2)
* **Mechanism:** `vk_queue::submit_sscp_kernel_from_code_object`,
  `submit_memcpy`, etc. each call `begin_command_buffer` /
  `submit_command_buffer`; each submit waits on the previous timeline value
  with `eAllCommands` (`src/runtime/vk/vk_queue.cpp`).
* **Impact:** 48 µs/launch (RADV), 35 µs/launch and ~57 futex/launch
  (llvmpipe) vs 7 µs HIP.
* **Proposed fix:** Record consecutive in-order operations into one command
  buffer separated by `vkCmdPipelineBarrier` (compute→compute,
  transfer↔compute); flush on `wait()`, host tasks, cross-queue deps,
  staging operations, or a batch-size/time threshold.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R2, R4 | M1 | confirms | 56% of RADV submit time in vkQueueSubmit |
| R6 | M1 | confirms | floydwarshall RADV 1.37× HIP, llvmpipe 5.6× OpenMP |
| R9, R10 | M2 | confirms | NVIDIA 17.6 µs host cost per launch vs 4.0 µs CUDA; llvmpipe ~79 futex per launch |
| R14 | M2 | confirms | bfs on a 1999-level grid (2 kernels + 2 one-byte copies + 3 waits per level): kernel time NVIDIA 2.0× CUDA, ANV 2.3–2.7× OpenCL; floydwarshall NVIDIA 1.27× |

#### <a id="impr-2"></a>IMPR-2: `get_command_buffer` queries the semaphore once per in-flight buffer
* **Status:** open
* **Mechanism:** The refresh loop in `vk_queue::get_command_buffer` calls
  `_semaphore.getCounterValue()` (an ioctl on RADV) inside the loop over
  `_executing_cmd_bufs`. The pool is fixed at `_cmd_buf_alloc_size = 24`
  (`vk_queue.hpp`), so the host blocks once 24 ops are in flight.
* **Impact:** 22% of RADV submit time; host stalls on llvmpipe.
* **Proposed fix:** Query the counter once before the loop; grow the pool
  on demand (allocate extra batches instead of blocking).
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R4 | M1 | confirms | |

#### <a id="impr-3"></a>IMPR-3: Host↔device copies slow (small 4-20×, 1-2 MB 10-50× vs HIP)
* **Status:** confirmed (M1, M2)
* **Mechanism:** Every copy to/from a host pointer:
  `vk_allocator::staging_allocate` (new VkBuffer + VkDeviceMemory) →
  GPU `copyBuffer` → `_host_worker` thread waits on the semaphore, maps,
  `memcpy`s, unmaps, signals → user thread wakes
  (`setup_staging_buffers` / `cleanup_staging_buffers`).
* **Impact:** D2H 4 B 151 µs vs 34 µs HIP; ~11 ioctl + mmap/munmap per copy.
  Hits any iterative algorithm with a convergence flag (e.g. bfs). Large
  copies are also 10-50× slower (R6), dominating end-to-end time of
  benchmarks that move results back to the host. Worst on the NVIDIA
  driver, where allocating+mapping+freeing a small staging buffer costs
  ~330 µs (R11): every copy takes ~570 µs (40× CUDA).
* **Proposed fix:**
  1. Pool and persistently map staging buffers. This is the fix that
     matters most on NVIDIA.
  2. If the user allocation is `HOST_VISIBLE` (all allocations on APUs,
     ReBAR on dGPUs), skip the GPU copy: wait for prior work, `memcpy`
     through a persistent mapping of the device allocation. **Only for
     H2D, or when the memory type is `HOST_CACHED`:** host reads through
     these mappings run at 0.02 GB/s (NVIDIA ReBAR) and 0.16 GB/s (ANV,
     write-combined), see R11 and [IMPR-13](#impr-13).
  3. Persistently map device allocations instead of map/unmap per use.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R2, R4 | M1 | confirms | 69% of time in the wait chain |
| R6 | M1 | confirms | memcpy: 64 B–64 KB RADV 117–159 µs vs HIP 5.5–20 µs; also hits **large** copies: 1–2 MB 0.46–2.6 ms vs 45–76 µs; mandelbrot 8 MB D2H makes "parallel time" 13× HIP; black-scholes transfers 1.8× |
| R11 | M2 | confirms | NVIDIA: 565–738 µs per small copy vs 9–15 µs CUDA; 2.1–3.0 GB/s vs 8.1–8.4 GB/s for 16–256 MB. Sampling: `staging_allocate` (`vkAllocateMemory`) on the submitting thread, `vkFreeMemory` in `cleanup_staging_buffers` on the worker; 10 ioctl + `openat`/`close` + mmap/munmap per copy |
| R11 | M2 | confirms | ANV: 179–235 µs per small copy vs 99–107 µs OpenCL; H2D 3.3–3.6 vs 11–13 GB/s (D2H: [IMPR-13](#impr-13)) |
| R11 | M2 | confirms | llvmpipe: ~4 GB/s for ≥1 MB vs 25–67 GB/s OpenCL CPU / 21–49 GB/s OpenMP |
| R14 | M2 | confirms | memcpy-sycl (pipelined): NVIDIA 7–117× CUDA, ANV 6–91× OpenCL, llvmpipe 1.4–14× OpenCL CPU; black-scholes processing time 2.0× (NVIDIA) / 3.7× (ANV); mandelbrot parallel time 4.2× / 18× |

#### <a id="impr-4"></a>IMPR-4: Per-launch uniform descriptor overhead
* **Status:** open (not measured in isolation; nbody args are push constants)
* **Mechanism:** `vk_kernel_pipeline::set_args` maps/unmaps uniform memory
  per argument and calls `updateDescriptorSets` per launch;
  `create_kernel_descriptors` calls `waitSemaphores` (ioctl) per slot to
  find a free one; only `MAX_INSTANCES = 10` slots per kernel
  (`vk_code_object.hpp/.cpp`).
* **Proposed fix:** Persistently mapped ring buffer for uniforms; track slot
  completion with one counter read; larger/growing slot count.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| code inspection | — | open | need a kernel with >256 B of args to measure |

#### <a id="impr-5"></a>IMPR-5: No FMA contraction (`NoContraction` on all float ops)
* **Status:** confirmed (M1, M2)
* **Mechanism:** `LLVMToCLSPV.cpp` invokes clspv without `--cl-mad-enable`;
  clspv then decorates every float op with `NoContraction`, ignoring LLVM
  `contract` fast-math flags present in the input IR.
* **Impact:** nbody RADV 690 → 765 GFLOP/s (+8-11%); llvmpipe ~+3%.
  NVIDIA: nbody accel kernel −15%, mandelbrot kernel −10% with
  `--cl-mad-enable`; no measurable effect on ANV, where other costs
  dominate (R16).
* **Proposed fix:** Pass `--cl-mad-enable` (or `--cl-unsafe-math-optimizations`
  under `-ffast-math`) when contraction is allowed by the source flags; or
  teach clspv to respect per-instruction `contract`. Keep a way to opt out
  for bit-reproducibility.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R1, R5 | M1 | confirms | verified by swapping clspv output into the JIT cache |
| R8 | M1 | confirms | also FP64: su3 has 12 `v_mul_f64` + 10 `v_add_f64` where HIP has 6 `v_fma_f64` |
| R13 | M2 | confirms | 64-step `x*a+b` chain: NVIDIA 1637 µs vs CUDA 1132 µs |
| R16 | M2 | confirms | via `clspv_shim.c`: NVIDIA nbody accel 7.72 → 6.55 ms (CUDA 5.42), mandelbrot kernel 1.15 → 1.04 ms (CUDA 0.71); ANV and llvmpipe unchanged within noise. IGC fuses the same loop into `mad` |
| R18 | M2 | confirms | no user-level workaround: `-ffast-math` changes nothing, `sycl::mad` is the same `NoContraction` multiply + add, and `sycl::fma` is emulated ([IMPR-16](#impr-16)). 1024-step `x*a+b` chain: NVIDIA 6.83 ms vs CUDA 2.77 ms |

#### <a id="impr-6"></a>IMPR-6: 64-bit index arithmetic (incl. 64-bit integer division)
* **Status:** confirmed (M1, M2)
* **Mechanism:** clspv invoked with `--arch=spir64`; `get_global_id` and
  indexing are 64-bit, producing `v_lshlrev_b64` + carry chains where HIP
  uses 32-bit `v_mad_u64_u32`. Worse, `size_t` division/modulo by a
  constant (very common for index decomposition) is emitted as a fully
  expanded 64-bit software division instead of a 32-bit magic-number
  multiply (R8). M2 adds two details:
  * Why the 32-bit narrowing that the CUDA path gets is missing (R12): the
    `GlobalSizesFitInI32OptPass` assumptions *are* applied, but the clspv
    libkernel implements `__acpp_sscp_get_group_id_*`/`get_local_id_*` as
    opaque i64 OpenCL calls (`src/libkernel/sscp/clspv/core.cpp`), so the
    global id is i64 from the start and InstCombine has no i32 `zext` to
    narrow. In the PTX flavour the same IR starts from i32 `ctaid`/`tid` and
    every `size_t` op in `idx.cpp` becomes i32. The `udiv i64` by a constant
    then reaches the driver as SPIR-V `OpUDiv %ulong`.
  * LLVM widens `int` loop counters that index memory to i64 (IndVarSimplify
    with the `spir64` data layout, which declares i64 native), and nothing
    in the clspv path strength-reduces the address computation (no LSR). On
    Xe-LPG, which has no 64-bit integer ALU, ANV then recomputes `p + j*40`
    with emulated 64-bit `mul`/`mach`/carry sequences every iteration of the
    nbody loop (~45 integer vs ~20 float instructions, R16).
* **Impact:** su3 RADV 2.6× slower than HIP (12.3 vs 32.3 GFLOP/s), almost
  entirely from 64-bit div/mod by 36/9/3. Small elsewhere, but in every
  kernel. `idx.cpp`: 15× slower than CUDA on NVIDIA, 2.5× slower than
  OpenCL on ANV; part of nbody's 2.5× on ANV.
* **Proposed fix:** Apply AdaptiveCpp's 32-bit global-id/range assumptions
  (as used for HIP) before handing IR to clspv so LLVM can narrow index
  maths; check why division by a constant isn't lowered to multiply-high
  in the clspv pipeline (possible `LegalizeIntWidthsPass` / clspv
  `udiv i64` lowering interaction). Concretely (from R12): have the clspv
  id builtins produce i32 values that are zero-extended (Vulkan's
  `gl_WorkGroupID`/`gl_LocalInvocationID` are 32-bit anyway), as the PTX
  builtins do; consider a data layout without native i64 (`n8:16:32`) or
  running LSR before clspv, so loop counters stay 32-bit; and lower any
  remaining `udiv`/`urem` by a constant to a multiply-high sequence in IR.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R5 | M1 | confirms (qualitative) | |
| R6, R8 | M1 | confirms | su3 2.6× slower; ISA dominated by 64-bit division expansion |
| R12 | M2 | confirms | `idx.cpp` `size_t`/`unsigned` time: NVIDIA 15.0 (247 vs 16.4 ms; CUDA 1.00), ANV 3.85 (OpenCL 1.62); SPIR-V has `OpUDiv %ulong ... %ulong_36`; IR comparison explains the missing narrowing |
| R16 | M2 | confirms | ANV nbody inner loop: 64-bit loop counter and per-iteration emulated 64-bit address maths |
| R14 | M2 | partial | su3: NVIDIA 1.34× CUDA, but ANV at parity with OpenCL (memory-bound there) |

#### <a id="impr-7"></a>IMPR-7: Read-only / no-alias information is dropped
* **Status:** open (speculative)
* **Summary:** `const`/`restrict` kernel pointers could be emitted with
  SPIR-V `NonWritable` / `RestrictPointer`, possibly letting RADV use SMEM
  (see FUND-2). Needs checking against what ACO actually consumes for PSB
  pointers.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|

#### <a id="impr-8"></a>IMPR-8: No VkPipelineCache
* **Status:** open (low priority)
* **Summary:** Pipelines are created with a null cache each process start;
  the driver's on-disk shader cache currently hides most of the cost.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|

#### <a id="impr-9"></a>IMPR-9 (related, not Vulkan): OpenMP backend slow on nbody `nd_range`
* **Status:** confirmed (M1, M2)
* **Summary:** The OpenMP backend takes 117 ms for the nbody accel kernel
  vs 46 ms on llvmpipe. Makes Vulkan look better in llvmpipe vs OpenMP
  comparisons; track separately. On M2 the Intel OpenCL CPU runtime runs the
  same kernel 3× faster than OpenMP, which makes it the better CPU reference.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R1, R2 | M1 | confirms | |
| R9, R14 | M2 | confirms | accel kernel: OpenMP 71.8 ms, llvmpipe 61.9 ms, OpenCL CPU 23.2 ms; HeCBench nbody 81 / 87 / 278 GFLOP/s. OpenMP also costs 41 µs per empty kernel on this hybrid 22-thread CPU |

#### <a id="impr-10"></a>IMPR-10: Atomics ~40% slower on RADV than HIP
* **Status:** confirmed (M1, M2); not root-caused
* **Summary:** histogram local-memory and global-memory atomic variants are
  1.42–1.44× slower on RADV, while atomicReduction (one global atomic per
  work-group after a reduction) is at parity.
* **To investigate:** compare ISA of the histogram kernels; check the clspv
  atomic lowering (`src/libkernel/sscp/clspv/atomic.cpp`, memory
  semantics/scope emitted — overly strong scopes or CAS loops instead of
  native `ds_add`/`global_atomic_add`). Not the barrier semantics of
  [IMPR-14](#impr-14): removing `UniformMemory` from histogram's barriers
  changes nothing (R16).
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R6 | M1 | confirms | smem 429 vs 297 µs; gmem 673 vs 473 µs |
| R14 | M2 | confirms | histogram first config: NVIDIA smem 134 vs 73 µs (1.84×), gmem 376 vs 338 µs (1.11×); ANV smem 713–1003 vs 647–696 µs (1.1–1.44×), gmem 910–1363 vs 412 µs (2.2–3.3×) vs OpenCL over two passes; atomicReduction at parity on both |

#### <a id="impr-11"></a>IMPR-11: Math builtins lowered to precise multi-instruction sequences
* **Status:** open
* **Mechanism:** `src/libkernel/sscp/clspv/math.cpp` maps
  `__acpp_sscp_rsqrt_f32` to OpenCL `rsqrt`, which clspv implements as
  `Sqrt` + denormal/overflow range scaling + `OpFDiv 1.0, x` + selects
  instead of GLSL `InverseSqrt`; only `sycl::native::rsqrt` reaches the
  hardware instruction. Every fp32 division gets the same range-scaling
  sequence (e.g. `2.0f / rows` in mandelbrot). On the CUDA backend all
  three spellings (`rsqrt`, `native::rsqrt`, `1/sqrt`) become
  `rsqrt.approx.f32` in the PTX, and IGC compiles OpenCL `rsqrt` to
  `math.rsqt`, so they cost the same there; part of the gap is therefore a
  difference in precision policy, not only in code quality.
* **Impact:** nbody accel on NVIDIA 7.6 ms with `sycl::rsqrt` vs 6.5 ms with
  `native::rsqrt` (CUDA 5.4 ms for both); HeCBench nbody uses `sycl::rsqrt`
  in its inner loop. Any kernel with `rsqrt` in hot loops; for divisions R18
  measured no gap against the native backends.
* **Proposed fix:** Map `rsqrt` to the native SPIR-V instruction (R18
  found no gap for division, see Evidence): Vulkan's precision for
  `InverseSqrt` (2 ULP) matches
  OpenCL's `rsqrt`, the difference is denormal/edge handling. Alternatively
  pass `--cl-native-math` / `--cl-fast-relaxed-math` /
  `--cl-denorms-are-zero` to clspv under `-ffast-math`. With
  `--cl-native-math` NVIDIA nbody accel drops from 7.72 to 5.95 ms.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R16 | M2 | confirms | `rsq.cpp`: NVIDIA rsqrt / native / 1-over-sqrt 7.58 / 6.54 / 7.57 ms (CUDA 5.42 for all); ANV 9.3–10.1 / 7.9–8.5 / 9.1–9.2 ms; llvmpipe 68 / 54–59 / 67 ms. SPIR-V inspection of `mb.cpp` and mandelbrot |
| R18 | M2 | partial | division is not a gap: a 128-step `a/(x+b)` chain is faster on Vulkan than natively (NVIDIA 4.13 vs CUDA 5.13 ms, ANV 3.49 vs OpenCL 4.01 ms). `native::recip` speeds it up on every GPU backend (Vulkan Ada 2.65, CUDA 4.03, Vulkan MTL 2.33, OpenCL MTL 2.13 ms); `native::divide` is broken on Vulkan ([BUG-15](#bug-15)) |

#### <a id="impr-12"></a>IMPR-12: No runtime loop unrolling or load/store vectorization in the clspv path
* **Status:** open
* **Mechanism:** AdaptiveCpp's own O3 pipeline is built without a target
  machine for every flavour (`withPassBuilderAndMAM`), but the PTX flavour
  then runs `opt --mcpu=sm_XX -O3` (NVPTX TTI, which enables runtime
  unrolling) and `llc -O3` (NVPTX runs LLVM's load/store vectorizer)
  (`src/compiler/llvm-to-backend/ptx/LLVMToPtx.cpp`); the clspv flavour
  hands the target-independent result to clspv, and Vulkan driver compilers
  do not runtime-unroll loops with unknown trip counts. The `single_task`
  sum in `mb.cpp` reaches CUDA unrolled ×8 with `ld.global.v4.b32`/`v2`
  loads, and reaches the Vulkan driver as a loop with one scalar `OpLoad`
  per iteration (and a 64-bit induction variable). Likewise nothing merges
  adjacent scalar accesses: copying nbody's 40-byte particle struct is 10
  scalar loads + 10 scalar stores in SPIR-V vs 5 `ld.global.v2` +
  4 `st.global.v2` in PTX.
* **Impact:** Latency-bound serial loops: NVIDIA `single_task` 805 vs
  226 µs (3.6×). Struct-heavy kernels: nbody update kernel 37 vs 19.5 µs of
  device time on NVIDIA. Possibly part of what FUND-2 attributes to scalar
  loads on M1.
* **Proposed fix:** Enable runtime/partial unrolling and the load/store
  vectorizer for the clspv flavour, e.g. by running `LoopUnrollPass` with
  runtime unrolling and `LoadStoreVectorizerPass` before handing IR to
  clspv (the equivalent of the PTX flavour's target-aware `opt`/`llc`
  step). Check that clspv and the drivers accept the resulting vector
  accesses on PSB pointers.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R9, R16 | M2 | confirms | PTX vs SPIR-V of the same kernels; for `single_task`, ANV (1658 µs) and OpenCL (1445 µs) are both slow, so it shows on NVIDIA only |
| R18 | M2 | confirms | `#pragma unroll 8` on a serial sum (stands in for the proposed fix): NVIDIA 3.15 → 1.14 ms (CUDA 0.91 ms without the pragma), ANV 6.41 → 2.56 ms (OpenCL 4.72 → 1.98 ms), llvmpipe 1.05 → 0.28 ms. The same pragma makes CUDA 2.7× slower (2.43 ms) |

#### <a id="impr-13"></a>IMPR-13: Staging buffers in uncached memory on ANV (D2H at 0.16 GB/s)
* **Status:** open
* **Mechanism:** `vk_allocator::staging_allocate` prefers
  `HOST_VISIBLE|HOST_COHERENT|HOST_CACHED` and falls back to
  `HOST_VISIBLE|HOST_COHERENT`. ANV (MTL) exposes coherent types that are
  uncached (write-combined) and cached types that are not coherent, so
  staging lands in write-combined memory and the host `memcpy` out of it in
  `cleanup_staging_buffers` reads at 0.16 GB/s (15 GB/s from the cached
  type, R11). The same applies to any future direct mapping of device
  allocations ([IMPR-3](#impr-3) fix 2): ANV's device allocations are
  write-combined and NVIDIA's ReBAR allocations read at 0.02 GB/s.
* **Impact:** Every D2H copy on ANV: 256 MB takes 1.66 s (OpenCL: 21 ms),
  1 MB 4.6 ms. Dominates mandelbrot's "parallel time" (45 vs 2.5 ms) and
  memcpy-sycl D2H (50–91× OpenCL at 1–2 MB).
* **Proposed fix:** Choose the staging memory type by direction: for D2H
  prefer `HOST_CACHED` even when it is not `HOST_COHERENT` and call
  `vkInvalidateMappedMemoryRanges` after the GPU copy; for H2D
  write-combined coherent memory is fine (20 GB/s writes).
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R11 | M2 | confirms | `xfer.cpp` D2H 0.16 GB/s from 16 MB up; `memtype.cpp` read bandwidth per memory type |
| R14 | M2 | confirms | memcpy-sycl D2H 1 MB 5.3 ms, 2 MB 9.6 ms vs 0.1 ms OpenCL |

#### <a id="impr-14"></a>IMPR-14: Work-group barriers always order global memory
* **Status:** open
* **Mechanism:** `sscp_barrier()`
  (`include/hipSYCL/sycl/libkernel/detail/device_barrier.hpp`) maps
  `fence_space::local_space` and `global_and_local` to the same
  `__acpp_sscp_work_group_barrier(work_group, seq_cst)`, and the clspv
  libkernel (`src/libkernel/sscp/clspv/barrier.cpp`) implements it as
  `barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)`. clspv emits
  `OpControlBarrier Workgroup Workgroup 0x148`
  (AcquireRelease|UniformMemory|WorkgroupMemory); the OpenCL SPIR-V path
  emits `0x110` (SequentiallyConsistent|WorkgroupMemory). ANV adds a
  global (UGM) fence before each such barrier; IGC only fences SLM.
* **Impact:** Barrier-heavy kernels using local memory. A/B by patching the
  cached SPIR-V to `0x108` (dropping `UniformMemory`): pathfinder NVIDIA
  3.45 → 2.18 s (CUDA 2.08 s), ANV 3.38 → 3.03 s; binomial NVIDIA
  10.1 → 7.1 ms (CUDA 5.2 ms), ANV 7.7 → 6.8 ms.
* **Proposed fix:** Pass the fence space (or an address-space mask) through
  the SSCP barrier builtin so that `local_space` barriers use
  `barrier(CLK_LOCAL_MEM_FENCE)`; keep global semantics for
  `global_space`/`global_and_local` and `group_barrier`.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R16 | M2 | confirms | `patch_barrier.sh` A/B above; binomial still passes; histogram unaffected |

#### <a id="impr-15"></a>IMPR-15: No reuse of device allocations
* **Status:** open
* **Mechanism:** Every `malloc_device` is a new `VkDeviceMemory`
  (`vk_allocator::raw_allocate`) and `free` releases it. On ANV the first
  kernel that touches a fresh allocation is slow (presumably the kernel
  driver populating and clearing the pages); on NVIDIA allocations
  themselves are expensive (119 µs for 4 KB of ReBAR memory, R11). The
  OpenCL backend on the same iGPU shows no first-use penalty.
* **Impact:** Code that allocates per iteration: mandelbrot kernel on ANV
  1.85–2.08 ms with a fresh 8 MB buffer vs 0.85–1.05 ms reused (OpenCL
  0.67–0.73 ms either way). Part of HeCBench mandelbrot's 3.5× on ANV.
* **Proposed fix:** Keep freed allocations in a size-bucketed cache (or
  sub-allocate from larger blocks) and hand them out again instead of
  calling `vkAllocateMemory`/`vkFreeMemory` per `malloc_device`/`free`.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R16 | M2 | confirms | `mandel.cpp` reused vs fresh buffer |

#### <a id="impr-16"></a>IMPR-16: `sycl::fma` is a software routine (40–73× slower than `a*b + c`)
* **Status:** open
* **Mechanism:** `__acpp_sscp_fma_f32` (`src/libkernel/sscp/clspv/math.cpp`)
  calls OpenCL `fma`. clspv lowers it to a correctly rounded software
  routine for finite inputs (~200 integer instructions; `GLSL.std.450 Fma`
  only on the NaN/Inf path). It emits a single `OpFmaKHR` (`SPV_KHR_fma`)
  only if `--spv-khr-fma` lists every float width clspv has enabled, and
  fp16 and fp64 are enabled by default (`lib/Compiler.cpp`). AdaptiveCpp
  passes no such option and does not enable `VK_KHR_shader_fma`.
* **Impact:** Every kernel that calls `sycl::fma` (common in code ported
  from CUDA's `fmaf`). 1024-step dependent chain: NVIDIA 499 ms vs 6.8 ms
  for `x*a+b` (CUDA 2.8 ms for both), ANV 351 vs 8.5 ms, llvmpipe 5.3 vs
  0.53 s. There is no fast user-level alternative today: `sycl::mad` and
  `a*b+c` are an unfused multiply and add ([IMPR-5](#impr-5)). On devices
  without `VK_KHR_shader_fma` the routine is unavoidable ([FUND-6](#fund-6)).
* **Proposed fix:** On devices with `VK_KHR_shader_fma`
  (`shaderFmaFloat16/32/64`; NVIDIA here, not ANV or llvmpipe with Mesa
  26.0.8): enable the extension and features at device creation, pass
  `--spv-khr-fma=<widths>` covering every width clspv has enabled (or
  disable unused widths), and accept capability `FMAKHR` (6030) in
  `verify_spv_capabilities`. With `--spv-khr-fma=16,32,64` via
  `clspv_shim.c` the chain takes 4.49 ms on NVIDIA. Devices without the
  extension still need the software routine for a correctly rounded `fma`,
  which SYCL 2020 requires ("Rounding of intermediate products shall not
  occur"): the Vulkan spec only guarantees `GLSL.std.450 Fma` the precision
  of "OpFMul followed by OpFAdd", whereas `OpFmaKHR` is "Correctly rounded".
  A relaxed mode (e.g. under `-ffast-math`) could map it to
  `GLSL.std.450 Fma` instead.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R18 | M2 | confirms | `uguide.cpp` chain on all three Vulkan drivers; SPIR-V inspection; `--spv-khr-fma` A/B on NVIDIA |

### C. Possible bugs

#### <a id="bug-1"></a>BUG-1: Profiling timestamps ignore `timestampPeriod`
* **Status:** confirmed (M1 RADV, M2 ANV)
* **Mechanism:** `vk_async_timestamp::get_time_point`
  (`include/hipSYCL/runtime/vk/vk_profiling.hpp`) treats raw query results
  as nanoseconds; they must be multiplied by
  `VkPhysicalDeviceLimits::timestampPeriod`. Start and finish timestamps
  each take their own calibration snapshot, so the reported duration is
  `raw ticks + (host time between the two snapshots)`; scaling
  `device_time - dev_ref` by `timestampPeriod` fixes both terms.
* **Impact:** On RADV (period 10) kernel durations are under-reported by
  exactly 10× (684 µs reported vs ~6.8 ms real). Any profiling-based
  comparison is wrong on such devices. ANV on MTL (period 52.08): ~44×
  (3.3 ms kernel reported as 75 µs).
* **Report:** [../bug-reports/BUG-1-vk-profiling-timestamp-period.md](../bug-reports/BUG-1-vk-profiling-timestamp-period.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R2 | M1 | confirms | |
| R13 | M2 | confirms | ANV wall 3285–3425 µs vs profiled 74.7–78.5 µs |
| R13 | M2 | n/a | NVIDIA and llvmpipe have period 1; their profiled times are correct |

#### <a id="bug-2"></a>BUG-2: Query pool destroyed while command buffer still executing
* **Status:** confirmed (M1, M2)
* **Mechanism:** Each profiled op owns a `vk::raii::QueryPool` inside its
  instrumentation object; if the user drops the event before completion,
  the pool is destroyed while still in use.
* **Repro:** `enable_profiling` + `in_order` queue, many fire-and-forget
  `parallel_for`s. Validation layer: `VUID-vkDestroyQueryPool-queryPool-00793`.
  llvmpipe: segfault in `libvulkan_lvp.so` queue thread. RADV:
  `The CS has been rejected` → `vk::DeviceLostError` (timing dependent).
  `tools/bug2.cpp` reproduces it.
* **Proposed fix:** Tie query pool lifetime to command buffer completion
  (e.g. keep a reference alongside `_executing_cmd_bufs`), or pool
  query slots per queue.
* **Report:** [../bug-reports/BUG-2-vk-query-pool-lifetime.md](../bug-reports/BUG-2-vk-query-pool-lifetime.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R2 | M1 | confirms | `mb.cpp` initial version with profiling queue |
| R13 | M2 | confirms | VUID on ANV, NVIDIA and llvmpipe; ANV and NVIDIA survive it, llvmpipe segfaults, aborts or hangs |

#### <a id="bug-3"></a>BUG-3: `ACPP_VISIBILITY_MASK` silently ignores unknown backend names
* **Status:** disputed (usability issue remains in a different form)
* **Summary:** `omp;vulkan` is accepted with no warning and the Vulkan
  backend isn't selected as intended. Should warn on unknown keys.
  On M2 (same commit), `vulkan` is an accepted alias for `vk` (since
  `98630f7a`) and selects the Vulkan backend. What remains: a misspelt
  backend name only prints a message and then the whole mask is discarded
  (all backends load), and a device filter that matches no device leaves
  the backend empty without a message. Both make it easy to benchmark the
  wrong device.
* **Report:** [../bug-reports/BUG-3-visibility-mask-invalid-entries.md](../bug-reports/BUG-3-visibility-mask-invalid-entries.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R1 | M1 | confirms | |
| R13 | M2 | refutes | `omp;vulkan` / `omp;vulkan:llvmpipe` work; see summary for the remaining problems |

#### <a id="bug-4"></a>BUG-4: `find_user_alloc` range check and linear scan
* **Status:** open (minor)
* **Mechanism:** `vk_allocator::find_user_alloc` uses `ptr > base` (the
  exact-base case is handled earlier so it is harmless) and does an O(n)
  scan for interior pointers on every memcpy/memset. Consider an ordered
  map with `upper_bound`.
* **Report:** [../bug-reports/BUG-4-vk-find-user-alloc-linear-scan.md](../bug-reports/BUG-4-vk-find-user-alloc-linear-scan.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| code inspection | — | — | |

#### <a id="bug-5"></a>BUG-5: HeCBench correctness failures (excluded from performance data)
* **Status:** open — each needs triage (benchmark bug vs AdaptiveCpp bug)
* **Summary:** Benchmarks/devices that produced incorrect results or
  failed to run. No performance data is recorded for them. Rows marked
  *Vulkan only* are the most likely Vulkan backend bugs. Distinct bugs are
  split out as BUG-6 to BUG-14, each with an upstream-ready report in
  [`../bug-reports/`](../bug-reports/).

| Benchmark (args) | Machine/Run | Failing devices | Symptom | Likely culprit |
|---|---|---|---|---|
| lavaMD (-boxes1d 10, `-DOUTPUT`) | M1/R7 | RADV, llvmpipe | max rel. error 45 (RADV), 9.9 (llvmpipe) vs OpenMP; HIP agrees to 2e-5 | **Vulkan only** → [BUG-8](#bug-8) |
| lud (-s 1024 -v) | M1/R7 | RADV, llvmpipe, OpenMP | mismatches from row 1; HIP passes | Vulkan + OpenMP (generic SSCP?) → [BUG-9](#bug-9) (Vulkan part) |
| layernorm (8 1024 12288 200) | M1/R6 | RADV | `Mismatch of out`, rc=1; HIP, OpenMP and llvmpipe report no mismatch | **Vulkan (RADV) only** |
| atomicCost (16 10) | M1/R6 | RADV, llvmpipe | FAIL on 2 of 3 sub-tests (incl. FP64 atomic add) | **Vulkan only** → [BUG-6](#bug-6) (FP64 part) |
| scan (16777216 100) | M1/R6 | RADV, llvmpipe | `LLVMToCLSPV: clspv invocation failed with exit code -2` (clspv crash), then hang until timeout | **Vulkan only** (clspv) → [BUG-11](#bug-11), [BUG-7](#bug-7) |
| scan | M1/R6 | OpenMP | segfault | OpenMP |
| shuffle (200000 100) | M1/R6 | OpenMP (all), llvmpipe (2 of 4) | FAIL | subgroup-size dependent (sg 8 on llvmpipe) |
| babelstream (default) | M1/R6 | HIP, RADV, OpenMP | dot check FAIL on all but llvmpipe | benchmark |
| pathfinder (`-DBENCH_PRINT`) | M1/R7 | OpenMP | output differs from HIP/RADV/llvmpipe | OpenMP |
| miniWeather | M1/R6 | — | not built (requires MPI) | n/a |
| lavaMD (-boxes1d 10, `-DOUTPUT`) | M2/R15 | ANV, NVIDIA, llvmpipe | max rel. error 42 (ANV), 21 (NVIDIA), 9.9 (llvmpipe, as on M1) vs CUDA; OpenCL GPU/CPU and OpenMP agree to ≤1.3e-5 | **Vulkan only** (all three drivers) → [BUG-8](#bug-8) |
| lud (-s 1024 -v) | M2/R15 | ANV, NVIDIA, llvmpipe, OpenMP; CUDA hangs; OpenCL CPU segfaults | Vulkan: ~32k mismatches from (1, 16); OpenMP: 2052 from (1, 0); OpenCL MTL passes | **Vulkan** (all drivers) + OpenMP → [BUG-9](#bug-9) |
| layernorm (8 1024 12288 200) | M2/R14 | ANV | `Mismatch of out`, rc=1; NVIDIA, CUDA and OpenCL MTL pass. Passes with the `vk/per-kernel-subgroup-size` backend (`055fe70d`, R15) | **Vulkan (ANV)**: sub-group size → [BUG-10](#bug-10) |
| softmax (100000 784 1 100) | M2/R14 | ANV; OpenCL MTL and CPU | ANV: wrong output (`device: 0.000000`); passes with `055fe70d` (12.9 ms). OpenCL: SIGFPE before any output | **Vulkan (ANV)**: sub-group size → [BUG-10](#bug-10); OpenCL crash → [BUG-14](#bug-14) |
| atomicCost (16 10) | M2/R14 | ANV (FP64 sub-test), llvmpipe (LLVM abort in the driver); OpenCL segfaults | needs `AtomicFloat64AddEXT`, which ANV and llvmpipe don't support; NVIDIA passes all 3 | **Vulkan backend**: [BUG-6](#bug-6) |
| scan (16777216 100) | M2/R14 | ANV, NVIDIA, llvmpipe | clspv `UNREACHABLE executed at lib/BitcastUtils.cpp:1443` ("Unexpected offset for type in GetIdxsForTyFromOffset", `SrcTy = [64 x i32]` / `[32 x i32]`, `DstTy = i8`); every further launch re-runs the failing JIT (~130 clspv crashes in 180 s), so the app looks hung | **Vulkan only** (clspv; retries: [BUG-7](#bug-7)) |
| scan | M2/R14 | OpenMP | segfault | OpenMP |
| shuffle (200000 100) | M2/R14 | ANV and OpenCL MTL (the sub-group-32 sub-tests), llvmpipe and OpenCL CPU (16 and 32), OpenMP (all) | identical pattern for both Intel GPU backends; `055fe70d` changes nothing | benchmark assumes sub-group sizes |
| babelstream (default) | M2/R14 | ANV, NVIDIA, CUDA (dot check); OpenCL segfaults | | benchmark (dot tolerance); OpenCL backend |
| histogram, atomicCost, babelstream | M2/R14 | OpenCL MTL and CPU | segfault once the runtime is re-initialised in the same process (`extracting function address for clEnqueueMemAdviseINTEL failed`, then `USM device allocation failed (CL:-32)`); histogram's first configuration is valid | OpenCL backend → [BUG-13](#bug-13) |
| binomial (default) | M2/R14 | llvmpipe | not a failure: >600 s (1000 launches × 4096 barriers, ~0.7 s each); the 20-launch variant passes | n/a (too slow) |
| nbody (16000 30) | M2/R14 | OpenCL CPU | `FAIL`: energy 172664 vs 172663, the tolerance artefact from R1 | benchmark |
| pathfinder (`-DBENCH_PRINT`) | M2/R15 | — | all 7 devices bit-identical, including OpenMP | — |

#### <a id="bug-6"></a>BUG-6: Unsupported SPIR-V capabilities are reported but not rejected
* **Status:** open
* **Mechanism:** `vk_executable_object::verify_spv_capabilities`
  (`src/runtime/vk/vk_code_object.cpp`) prints
  `unexpected SPIRV capability <n>` for capabilities it doesn't know and
  carries on. `AtomicFloat64AddEXT` (6034, from `atomic_ref<double>` adds) is
  one of them; `VK_EXT_shader_atomic_float` is never enabled or queried.
* **Impact:** On devices without `shaderBufferFloat64AtomicAdd` (ANV MTL,
  llvmpipe; probably RADV on M1) the kernel runs anyway: wrong results on
  ANV, an LLVM instruction-selection abort inside llvmpipe
  (`Cannot select: ... store<(store (s64)), trunc to f64>`). NVIDIA
  supports the feature and passes.
* **Proposed fix:** Enable `VK_EXT_shader_atomic_float` features when
  present and treat unknown/unsupported capabilities as a build error; for
  missing float atomics, emulate with a CAS loop in the clspv libkernel (as
  the Metal backend does for 64-bit atomics).
* **Report:** [../bug-reports/BUG-6-vk-unsupported-spirv-capabilities.md](../bug-reports/BUG-6-vk-unsupported-spirv-capabilities.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R14, R15 | M2 | confirms | atomicCost FP64 sub-test; `vulkaninfo`: `shaderBufferFloat64AtomicAdd` false on ANV and llvmpipe, true on NVIDIA |

#### <a id="bug-7"></a>BUG-7: Failed JIT compilations are retried on every launch
* **Status:** open
* **Mechanism:** When clspv fails, `submit_sscp_kernel_from_code_object`
  returns "Code object construction failed", the node is cancelled and the
  error is registered asynchronously, but nothing records the failure in
  the kernel cache: `kernel_cache::get_or_construct_jit_code_object`
  (`include/hipSYCL/runtime/kernel_cache.hpp`, shared by all JIT backends)
  returns `nullptr` without caching anything. Every later launch of the same
  kernel runs the whole JIT (and clspv) again. Errors only surface through
  async handlers, so an application that uses `wait()` keeps going.
* **Impact:** scan-sycl, which M1 reported as "hang until timeout", spends
  its time in ~130 failed clspv runs of ~1.4 s each in 180 s; the same would
  happen for any kernel the Vulkan path can't compile.
* **Proposed fix:** Cache compilation failures per binary configuration
  and fail fast (or throw synchronously from the submission) on later
  launches.
* **Report:** [../bug-reports/BUG-7-sscp-failed-jit-retried-every-launch.md](../bug-reports/BUG-7-sscp-failed-jit-retried-every-launch.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R14, R15 | M2 | confirms | scan logs: 129–130 failed JIT compilations (`UNREACHABLE` in clspv → `Code object construction failed`) per 180 s run on each Vulkan driver |
| R17 | M2 | confirms | minimal reproducer: 3 launches, 3 clspv runs of ~1.74 s, kernel never runs |

#### <a id="bug-8"></a>BUG-8: HeCBench lavaMD gives wrong results on every Vulkan driver
* **Status:** confirmed (M1, M2); not reduced
* **Summary:** Split out of BUG-5. Max. relative error against a native
  backend: RADV 45, llvmpipe 9.9 (M1); ANV 42, NVIDIA 21, llvmpipe 9.9 (M2).
  OpenCL (GPU and CPU), OpenMP, HIP and CUDA agree to ~1e-5.
* **Report:** [../bug-reports/BUG-8-vk-lavamd-wrong-results.md](../bug-reports/BUG-8-vk-lavamd-wrong-results.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R7 | M1 | confirms | RADV, llvmpipe |
| R15 | M2 | confirms | ANV, NVIDIA, llvmpipe |

#### <a id="bug-9"></a>BUG-9: HeCBench lud gives wrong results on every Vulkan driver
* **Status:** confirmed (M1, M2); not reduced
* **Summary:** Split out of BUG-5. ~32k mismatches with `-s 1024 -v` on ANV,
  NVIDIA and llvmpipe (M2) and on RADV and llvmpipe (M1); OpenCL on the same
  Intel GPU and HIP pass. The first mismatch is at (1, 16) on the GPUs and
  (1, 8) on llvmpipe. OpenMP also fails, with a different pattern.
* **Report:** [../bug-reports/BUG-9-vk-lud-wrong-results.md](../bug-reports/BUG-9-vk-lud-wrong-results.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R7 | M1 | confirms | RADV, llvmpipe |
| R15 | M2 | confirms | ANV, NVIDIA, llvmpipe |

#### <a id="bug-10"></a>BUG-10: Intel ANV runs kernels with a different sub-group size than reported
* **Status:** open; fix on branch `vk/per-kernel-subgroup-size` (`055fe70d`)
* **Summary:** Split out of BUG-5. The backend reports 32 but ANV compiles
  SIMD16 unless a size is required; sub-group algorithms and queries break
  (softmax, layernorm). Not checked whether M1's RADV layernorm failure has
  the same cause.
* **Report:** [../bug-reports/BUG-10-vk-anv-subgroup-size.md](../bug-reports/BUG-10-vk-anv-subgroup-size.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R14, R15, R17 | M2 | confirms | softmax/layernorm fail at `f4ff3681`, pass with `055fe70d`; reproducer SIMD16 vs SIMD32 |

#### <a id="bug-11"></a>BUG-11: clspv aborts on unaligned byte GEPs into local memory (scan)
* **Status:** confirmed (M1, M2); reduced
* **Summary:** Split out of BUG-5. AdaptiveCpp's dynamic local memory is an
  `[N x i32]` global; `char` accesses at byte offsets that are not a
  multiple of 4 make clspv's `SimplifyPointerBitcastPass` hit `UNREACHABLE` in
  `BitcastUtils::GetIdxsForTyFromOffset`. Reproducers for AdaptiveCpp (SYCL)
  and clspv (`clspv-opt` on 8 lines of IR) are in the report.
* **Report:** [../bug-reports/BUG-11-vk-clspv-unaligned-local-gep.md](../bug-reports/BUG-11-vk-clspv-unaligned-local-gep.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R6 | M1 | confirms | scan: clspv exit code -2 on RADV and llvmpipe |
| R14, R17 | M2 | confirms | all three drivers; reduced and reproduced with `clspv-opt` |

#### <a id="bug-12"></a>BUG-12 (related, not Vulkan): OpenCL accepts `enable_profiling` without profiling support
* **Status:** open
* **Summary:** OpenCL devices report `aspect::queue_profiling == false` (the
  backend has no profiling code), but the queue constructor does not throw
  `errc::feature_not_supported` as SYCL 2020 requires;
  `get_profiling_info` aborts later.
* **Report:** [../bug-reports/BUG-12-ocl-enable-profiling-not-rejected.md](../bug-reports/BUG-12-ocl-enable-profiling-not-rejected.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R13, R17 | M2 | confirms | Intel GPU and CPU OpenCL |

#### <a id="bug-13"></a>BUG-13 (related, not Vulkan): OpenCL USM fails after runtime re-initialisation
* **Status:** open
* **Summary:** After the runtime shuts down and restarts in the same process,
  `clGetExtensionFunctionAddressForPlatform` returns null for all USM entry
  points and `malloc_device` fails with `CL:-32`. Breaks HeCBench histogram,
  atomicCost and babelstream on OpenCL after their first configuration.
* **Report:** [../bug-reports/BUG-13-ocl-usm-after-runtime-reinit.md](../bug-reports/BUG-13-ocl-usm-after-runtime-reinit.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R14, R17 | M2 | confirms | 15-line reproducer; Vulkan unaffected |

#### <a id="bug-14"></a>BUG-14 (related, not Vulkan): OpenCL `sub_group_sizes` heuristic
* **Status:** open
* **Summary:** The OpenCL backend reports every power of two up to the
  work-group size (1…512 on the Intel GPU) instead of the device's sizes
  (8, 16, 32). HeCBench softmax divides by the largest value and crashes with
  SIGFPE on both OpenCL devices.
* **Report:** [../bug-reports/BUG-14-ocl-sub-group-sizes.md](../bug-reports/BUG-14-ocl-sub-group-sizes.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R14, R17 | M2 | confirms | Intel GPU and CPU OpenCL |

#### <a id="bug-15"></a>BUG-15: `sycl::native::divide` makes the kernel fail to load
* **Status:** open
* **Summary:** Any kernel that calls `sycl::native::divide` fails on the
  Vulkan backend with `failed to parse spirv` and `Unknown kernel name`, and
  the program terminates. clspv receives a one-operand call
  (`_Z13native_dividef`), exits successfully, and emits an `OpFDiv` whose
  second operand id is 0 (`spirv-val`: `Id is 0`). The clspv libkernel
  declares `float native_divide(float)` and calls `native_divide(x)`
  (`src/libkernel/sscp/clspv/native.cpp`, unchanged on `develop` on
  2026-10-30). Editing the dumped IR to pass both operands gives a valid
  module. The other `sycl::native` functions tested work.
* **Impact:** Kernels that use it do not run on Vulkan. Workaround:
  `a * sycl::native::recip(b)`, which works and is the faster spelling on
  every GPU backend in R18.
* **Report:** [../bug-reports/BUG-15-vk-native-divide.md](../bug-reports/BUG-15-vk-native-divide.md)
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R18 | M2 | confirms | ANV, NVIDIA and llvmpipe; OpenCL GPU/CPU, OpenMP and CUDA correct |

---

## Benchmark plan

HeCBench benchmarks proposed for follow-up runs, with the findings each is
expected to exercise. Tick off as runs are added.

| Benchmark | Targets | Done (runs) |
|-----------|---------|-------------|
| nbody-sycl | IMPR-5, FUND-2, baseline | R1 (M1), R14/R16 (M2) |
| bfs-sycl | IMPR-3, IMPR-1 (1-byte H2D/D2H + 2 kernels per iteration) | R6 (M1) — random graph has few levels; a deeper graph would stress IMPR-3 more. R14 (M2): random graph plus a 1000×1000 grid (1999 levels, `gen_bfs_graph.py`) |
| floydwarshall-sycl, gaussian-sycl, pathfinder-sycl | FUND-1, IMPR-1, IMPR-2 | R6 (M1), R14 (M2); pathfinder also IMPR-14 (R16) |
| lud-sycl | FUND-1, IMPR-1 | excluded (BUG-5, M1 and M2) |
| memcpy-sycl | IMPR-3 | R6 (M1), R14 (M2) |
| babelstream-sycl | FUND-3 | excluded (BUG-5) |
| black-scholes-sycl, binomial-sycl, mandelbrot-sycl | IMPR-5, clspv math lowering | R6 (M1), R14/R16 (M2): IMPR-11, IMPR-14, IMPR-15 |
| softmax-sycl | subgroups | R6 (M1), R14 (M2: NVIDIA only; ANV needs the sub-group fix, OpenCL crashes) |
| scan-sycl, layernorm-sycl, shuffle-sycl | local mem, barriers, subgroups (sg 8 vs 64) | excluded (BUG-5) |
| histogram-sycl, atomicReduction-sycl | atomics lowering | R6 (M1), R14 (M2) |
| atomicCost-sycl | atomics lowering | excluded (BUG-5, BUG-6) |
| su3-sycl | general codegen | R6, R8 (M1), R14 (M2) |
| lavaMD-sycl | general codegen | excluded (BUG-5, M1 and M2) |
| miniWeather-sycl | general codegen | not built (MPI) |

Suggested additions based on R6–R8: an integer-heavy index-decomposition
kernel to isolate IMPR-6; a memcpy sweep beyond 2 MB for IMPR-3;
ISA comparison for histogram (IMPR-10). The first two are done
(`idx.cpp` R12, `xfer.cpp` R11).

Follow-ups suggested by R9–R16:
* Validate the proposed fixes with patched builds: i32 id builtins and
  32-bit loop counters for clspv (IMPR-6), runtime unrolling and load/store
  vectorization (IMPR-12), fence-space-aware barriers (IMPR-14), staging
  memory type by direction (IMPR-13), staging/allocation pools (IMPR-3,
  IMPR-15).
* Re-run the ANV numbers with `vk/per-kernel-subgroup-size` merged:
  forcing SIMD32 may change performance as well as correctness (softmax
  12.9 ms with it, no native comparison available).
* Root-cause lavaMD and lud on Vulkan (wrong on every Vulkan driver, so
  likely clspv-path codegen) and the remaining ~1.5× of mandelbrot's loop
  on NVIDIA.
* ISA comparison for histogram (IMPR-10) on ANV vs IGC.
* Check on M1 whether HIP's `single_task` loop is unrolled (IMPR-12 vs
  FUND-2).

Machines that would add the most information: a discrete AMD GPU (non-APU
memory types), NVIDIA (proprietary Vulkan vs CUDA), Intel (ANV vs L0), and
a mobile/Adreno device. M2 covers NVIDIA and Intel (vs OpenCL; no Level
Zero backend was built); discrete AMD, Intel vs L0 and mobile remain.
