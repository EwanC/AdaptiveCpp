# Vulkan backend performance analysis

Living document tracking performance of the AdaptiveCpp Vulkan backend
compared with other backends targeting the *same* hardware:

* Vulkan GPU driver (e.g. RADV) vs the native GPU backend (HIP / CUDA / L0)
* Vulkan CPU driver (llvmpipe/lavapipe) vs the OpenMP backend

The document is designed to be extended from multiple machines. Every finding
has a stable ID; new benchmark runs are appended to the [run log](#run-log)
and then referenced from the finding's **Evidence** table with a verdict.

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

* `ACPP_VISIBILITY_MASK` backend key is **`vk`**, not `vulkan`. Unknown keys
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

### Tools

Sources in [`tools/`](tools/). Build each with
`acpp -O3 --acpp-targets=generic <file>.cpp -o <name>`.

| Tool | Purpose |
|------|---------|
| `mb.cpp [n]` | Launch latency, small memcpy latency, nbody kernels timed individually (wall clock) |
| `bw.cpp <MB>` | STREAM-like `b[i]=2*a[i]+1` bandwidth & per-kernel cost for back-to-back dependent kernels |
| `lat.cpp <K> <M>` | K empty kernel launches, then M 4-byte D2H copies; target for `strace -f -c` and sampling |
| `samp.sh` / `samp.gdb` / `agg.py` | gdb-based stack sampler for when `perf` / ptrace-attach is unavailable |
| `hecbench_run.sh` | Build instructions + runner for a HeCBench SYCL benchmark on all comparison devices (warm-up + measured run) |

### Correctness gate for HeCBench data

Only record performance data for a benchmark/device pair if the result is
verified correct. Many HeCBench benchmarks only self-check optionally (e.g.
`lud -v`) or not at all (e.g. `pathfinder` needs `-DBENCH_PRINT`, `lavaMD`
needs `-DOUTPUT`); cross-check their outputs between devices before trusting
timings. Failures go in [BUG-5](#bug-5) without performance data.

Useful external tools: `RADV_DEBUG=shaders,nocache` (dumps ACO ISA),
`spirv-dis` on JIT cache entries in `~/.acpp/apps/<app>/jit-cache/*.jit`,
`llvm-objdump -d --mcpu=<gfx>` on HIP JIT cache ELFs,
`ACPP_S2_DUMP_IR_FINAL=1` (dumps the LLVM IR handed to clspv),
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`.

---

## Machines

### M1
* CPU/GPU: AMD Ryzen 7 PRO 8840HS, Radeon 780M (gfx1103, RDNA3 APU), 16 threads
* OS: Linux 6.17.0-1028-oem (Ubuntu 24.04)
* Vulkan: Mesa 25.2.8 (RADV PHOENIX; llvmpipe LLVM 20.1.2, 256-bit), Vulkan SDK 1.4.321.1
* RADV `timestampPeriod` = 10, subgroup size 64; llvmpipe `timestampPeriod` = 1, subgroup size 8
* Memory: every RADV device allocation lands in a `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` type (APU)
* AdaptiveCpp: `f4ff3681` (Release build), clspv per `doc/install-vulkan.md`

---

## Run log

### R1 — M1, nbody-sycl (HeCBench), 2026-10-29
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

### R2 — M1, `tools/mb.cpp` (n=16000), 2026-10-29
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

### R3 — M1, `tools/bw.cpp`, 2026-10-29
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

### R4 — M1, syscalls / sampling, 2026-10-29
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

### R5 — M1, codegen inspection, 2026-10-29
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

### R6 — M1, HeCBench suite, 2026-10-29
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

### R7 — M1, HeCBench correctness triage, 2026-10-29
Correctness of every pair in R6 was confirmed by the benchmark's own check
(bfs, floydwarshall, gaussian, binomial, mandelbrot, softmax, histogram,
atomicReduction, su3 via `assert`), by comparing printed results
(black-scholes output price at index 25000000 agrees to 6 digits on all
devices), or by building a print variant and diffing between devices
(pathfinder with `-DBENCH_PRINT`, 100000 100 5: HIP, RADV and llvmpipe
bit-identical; **OpenMP differs**, so pathfinder OpenMP data is excluded).
Benchmarks excluded entirely are listed in [BUG-5](#bug-5).

### R8 — M1, su3 ISA inspection, 2026-10-29
Kernel computes `size_t i = id/36; j = (id%36)/9; k = (id%9)/3; l = id%3`.

| | HIP | RADV |
|---|---|---|
| Integer div/mod | 32-bit magic-number multiply (`v_mad_u64_u32` ×5) | fully expanded 64-bit division: 125 `v_cmp_le_u64`, 334 `v_cndmask_b32`, 137 `v_add_co_u32`, 27 `s_bfm_b64` |
| FP64 complex multiply-add | 6 `v_fma_f64` | 12 `v_mul_f64` + 10 `v_add_f64` (no FMA) |
| Memory | 6 `global_load_b128` | 6 `global_load_b128` |

---

## Findings

Finding format: **Status**, **Summary**, **Mechanism** (code pointers),
**Impact**, **User guidance** (fundamental only) or **Proposed fix**
(improvable/bugs), **Evidence** table.

### A. Fundamental limitations (communicate to users)

Things that are inherent to Vulkan, its SPIR-V environment, or the drivers,
and which AdaptiveCpp can mitigate but not eliminate.

#### FUND-1: Per-submission cost and kernel-resolved dependencies
* **Status:** open
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

#### FUND-2: No scalar (SMEM) loads for wave-uniform pointer loads on RADV
* **Status:** open
* **Summary:** HIP's backend proves uniform loads are unclobbered and uses
  the scalar cache. Through SPIR-V `PhysicalStorageBuffer` pointers the
  Vulkan driver compiler has no alias/read-only information and emits vector
  memory loads.
* **Impact:** Serial `single_task` loops (1.68 vs 1.17 ms) and uniform
  reads such as `p[j]` in nbody. Largest remaining nbody gap after IMPR-5.
* **User guidance:** Avoid serial `single_task` reductions; use
  `sycl::reduction` or group algorithms.
* **Note:** partially addressable via IMPR-7; may require driver work.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R2, R5 | M1 | confirms | |

#### FUND-3: llvmpipe global-memory bandwidth roughly half of OpenMP
* **Status:** open (mechanism unconfirmed)
* **Summary:** Streaming kernels on llvmpipe reach ~16 GB/s vs ~32 GB/s for
  OpenMP; more threads do not help. Hypothesis: lavapipe lowers
  physical-storage-buffer accesses to per-lane gathers/scatters instead of
  contiguous vector loads.
* **Impact:** Bandwidth-bound kernels on llvmpipe. Compute-bound kernels
  are fine (nbody accel is 2.5× faster than OpenMP).
* **User guidance:** llvmpipe is suited to compute-bound kernels.
* **To verify:** dump lavapipe LLVM IR (`LP_DEBUG`/`GALLIVM_DEBUG`) for
  `bw.cpp`.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R3 | M1 | confirms | 64-256 MB: 16-17 GB/s vs 27-32 GB/s |
| R6 | M1 | partial | atomicReduction (bandwidth-bound) 20.7 vs 39.5 GB/s best case; but llvmpipe is also slow on non-bandwidth-bound kernels, see FUND-5 |

#### FUND-4: No host / shared USM
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

#### FUND-5: llvmpipe slower than OpenMP on most real kernels
* **Status:** open (mechanism unknown)
* **Summary:** Although llvmpipe beats OpenMP on nbody (compute-bound,
  regular), it is 1.8–5.6× slower on most HeCBench kernels, including ones
  with no barriers, local memory or atomics (floydwarshall, gaussian, bfs,
  black-scholes). Exceptions: mandelbrot (parity), su3 (faster, as OpenMP
  is slow there).
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

### B. Improvable performance (actionable in AdaptiveCpp)

Ordered roughly by expected impact on M1.

#### IMPR-1: One command buffer + one vkQueueSubmit per operation
* **Status:** open
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

#### IMPR-2: `get_command_buffer` queries the semaphore once per in-flight buffer
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

#### IMPR-3: Host↔device copies slow (small 4-20×, 1-2 MB 10-50× vs HIP)
* **Status:** open
* **Mechanism:** Every copy to/from a host pointer:
  `vk_allocator::staging_allocate` (new VkBuffer + VkDeviceMemory) →
  GPU `copyBuffer` → `_host_worker` thread waits on the semaphore, maps,
  `memcpy`s, unmaps, signals → user thread wakes
  (`setup_staging_buffers` / `cleanup_staging_buffers`).
* **Impact:** D2H 4 B 151 µs vs 34 µs HIP; ~11 ioctl + mmap/munmap per copy.
  Hits any iterative algorithm with a convergence flag (e.g. bfs). Large
  copies are also 10-50× slower (R6), dominating end-to-end time of
  benchmarks that move results back to the host.
* **Proposed fix:**
  1. Pool and persistently map staging buffers.
  2. If the user allocation is `HOST_VISIBLE` (all allocations on APUs,
     ReBAR on dGPUs), skip the GPU copy: wait for prior work, `memcpy`
     through a persistent mapping of the device allocation.
  3. Persistently map device allocations instead of map/unmap per use.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R2, R4 | M1 | confirms | 69% of time in the wait chain |
| R6 | M1 | confirms | memcpy: 64 B–64 KB RADV 117–159 µs vs HIP 5.5–20 µs; also hits **large** copies: 1–2 MB 0.46–2.6 ms vs 45–76 µs; mandelbrot 8 MB D2H makes "parallel time" 13× HIP; black-scholes transfers 1.8× |

#### IMPR-4: Per-launch uniform descriptor overhead
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

#### IMPR-5: No FMA contraction (`NoContraction` on all float ops)
* **Status:** open
* **Mechanism:** `LLVMToCLSPV.cpp` invokes clspv without `--cl-mad-enable`;
  clspv then decorates every float op with `NoContraction`, ignoring LLVM
  `contract` fast-math flags present in the input IR.
* **Impact:** nbody RADV 690 → 765 GFLOP/s (+8-11%); llvmpipe ~+3%.
* **Proposed fix:** Pass `--cl-mad-enable` (or `--cl-unsafe-math-optimizations`
  under `-ffast-math`) when contraction is allowed by the source flags; or
  teach clspv to respect per-instruction `contract`. Keep a way to opt out
  for bit-reproducibility.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R1, R5 | M1 | confirms | verified by swapping clspv output into the JIT cache |
| R8 | M1 | confirms | also FP64: su3 has 12 `v_mul_f64` + 10 `v_add_f64` where HIP has 6 `v_fma_f64` |

#### IMPR-6: 64-bit index arithmetic (incl. 64-bit integer division)
* **Status:** open
* **Mechanism:** clspv invoked with `--arch=spir64`; `get_global_id` and
  indexing are 64-bit, producing `v_lshlrev_b64` + carry chains where HIP
  uses 32-bit `v_mad_u64_u32`. Worse, `size_t` division/modulo by a
  constant (very common for index decomposition) is emitted as a fully
  expanded 64-bit software division instead of a 32-bit magic-number
  multiply (R8).
* **Impact:** su3 RADV 2.6× slower than HIP (12.3 vs 32.3 GFLOP/s), almost
  entirely from 64-bit div/mod by 36/9/3. Small elsewhere, but in every
  kernel.
* **Proposed fix:** Apply AdaptiveCpp's 32-bit global-id/range assumptions
  (as used for HIP) before handing IR to clspv so LLVM can narrow index
  maths; check why division by a constant isn't lowered to multiply-high
  in the clspv pipeline (possible `LegalizeIntWidthsPass` / clspv
  `udiv i64` lowering interaction).
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R5 | M1 | confirms (qualitative) | |
| R6, R8 | M1 | confirms | su3 2.6× slower; ISA dominated by 64-bit division expansion |

#### IMPR-7: Read-only / no-alias information is dropped
* **Status:** open (speculative)
* **Summary:** `const`/`restrict` kernel pointers could be emitted with
  SPIR-V `NonWritable` / `RestrictPointer`, possibly letting RADV use SMEM
  (see FUND-2). Needs checking against what ACO actually consumes for PSB
  pointers.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|

#### IMPR-8: No VkPipelineCache
* **Status:** open (low priority)
* **Summary:** Pipelines are created with a null cache each process start;
  the driver's on-disk shader cache currently hides most of the cost.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|

#### IMPR-9 (related, not Vulkan): OpenMP backend slow on nbody `nd_range`
* **Status:** open
* **Summary:** The OpenMP backend takes 117 ms for the nbody accel kernel
  vs 46 ms on llvmpipe. Makes Vulkan look better in llvmpipe vs OpenMP
  comparisons; track separately.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R1, R2 | M1 | confirms | |

#### IMPR-10: Atomics ~40% slower on RADV than HIP
* **Status:** open (not root-caused)
* **Summary:** histogram local-memory and global-memory atomic variants are
  1.42–1.44× slower on RADV, while atomicReduction (one global atomic per
  work-group after a reduction) is at parity.
* **To investigate:** compare ISA of the histogram kernels; check the clspv
  atomic lowering (`src/libkernel/sscp/clspv/atomic.cpp`, memory
  semantics/scope emitted — overly strong scopes or CAS loops instead of
  native `ds_add`/`global_atomic_add`).
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R6 | M1 | confirms | smem 429 vs 297 µs; gmem 673 vs 473 µs |

### C. Possible bugs

#### BUG-1: Profiling timestamps ignore `timestampPeriod`
* **Status:** open
* **Mechanism:** `vk_async_timestamp::get_time_point`
  (`include/hipSYCL/runtime/vk/vk_profiling.hpp`) treats raw query results
  as nanoseconds; they must be multiplied by
  `VkPhysicalDeviceLimits::timestampPeriod`.
* **Impact:** On RADV (period 10) kernel durations are under-reported by
  exactly 10× (684 µs reported vs ~6.8 ms real). Any profiling-based
  comparison is wrong on such devices.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R2 | M1 | confirms | |

#### BUG-2: Query pool destroyed while command buffer still executing
* **Status:** open
* **Mechanism:** Each profiled op owns a `vk::raii::QueryPool` inside its
  instrumentation object; if the user drops the event before completion,
  the pool is destroyed while still in use.
* **Repro:** `enable_profiling` + `in_order` queue, many fire-and-forget
  `parallel_for`s. Validation layer: `VUID-vkDestroyQueryPool-queryPool-00793`.
  llvmpipe: segfault in `libvulkan_lvp.so` queue thread. RADV:
  `The CS has been rejected` → `vk::DeviceLostError` (timing dependent).
* **Proposed fix:** Tie query pool lifetime to command buffer completion
  (e.g. keep a reference alongside `_executing_cmd_bufs`), or pool
  query slots per queue.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R2 | M1 | confirms | `mb.cpp` initial version with profiling queue |

#### BUG-3: `ACPP_VISIBILITY_MASK` silently ignores unknown backend names
* **Status:** open (usability)
* **Summary:** `omp;vulkan` is accepted with no warning and the Vulkan
  backend isn't selected as intended. Should warn on unknown keys.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| R1 | M1 | confirms | |

#### BUG-4: `find_user_alloc` range check and linear scan
* **Status:** open (minor)
* **Mechanism:** `vk_allocator::find_user_alloc` uses `ptr > base` (the
  exact-base case is handled earlier so it is harmless) and does an O(n)
  scan for interior pointers on every memcpy/memset. Consider an ordered
  map with `upper_bound`.
* **Evidence:**

| Run | Machine | Verdict | Notes |
|-----|---------|---------|-------|
| code inspection | — | — | |

#### BUG-5: HeCBench correctness failures (excluded from performance data)
* **Status:** open — each needs triage (benchmark bug vs AdaptiveCpp bug)
* **Summary:** Benchmarks/devices that produced incorrect results or
  failed to run. No performance data is recorded for them. Rows marked
  *Vulkan only* are the most likely Vulkan backend bugs.

| Benchmark (args) | Machine/Run | Failing devices | Symptom | Likely culprit |
|---|---|---|---|---|
| lavaMD (-boxes1d 10, `-DOUTPUT`) | M1/R7 | RADV, llvmpipe | max rel. error 45 (RADV), 9.9 (llvmpipe) vs OpenMP; HIP agrees to 2e-5 | **Vulkan only** |
| lud (-s 1024 -v) | M1/R7 | RADV, llvmpipe, OpenMP | mismatches from row 1; HIP passes | Vulkan + OpenMP (generic SSCP?) |
| layernorm (8 1024 12288 200) | M1/R6 | RADV | `Mismatch of out`, rc=1; HIP, OpenMP and llvmpipe report no mismatch | **Vulkan (RADV) only** |
| atomicCost (16 10) | M1/R6 | RADV, llvmpipe | FAIL on 2 of 3 sub-tests (incl. FP64 atomic add) | **Vulkan only** |
| scan (16777216 100) | M1/R6 | RADV, llvmpipe | `LLVMToCLSPV: clspv invocation failed with exit code -2` (clspv crash), then hang until timeout | **Vulkan only** (clspv) |
| scan | M1/R6 | OpenMP | segfault | OpenMP |
| shuffle (200000 100) | M1/R6 | OpenMP (all), llvmpipe (2 of 4) | FAIL | subgroup-size dependent (sg 8 on llvmpipe) |
| babelstream (default) | M1/R6 | HIP, RADV, OpenMP | dot check FAIL on all but llvmpipe | benchmark |
| pathfinder (`-DBENCH_PRINT`) | M1/R7 | OpenMP | output differs from HIP/RADV/llvmpipe | OpenMP |
| miniWeather | M1/R6 | — | not built (requires MPI) | n/a |

---

## Benchmark plan

HeCBench benchmarks proposed for follow-up runs, with the findings each is
expected to exercise. Tick off as runs are added.

| Benchmark | Targets | Done (runs) |
|-----------|---------|-------------|
| nbody-sycl | IMPR-5, FUND-2, baseline | R1 (M1) |
| bfs-sycl | IMPR-3, IMPR-1 (1-byte H2D/D2H + 2 kernels per iteration) | R6 (M1) — random graph has few levels; a deeper graph would stress IMPR-3 more |
| floydwarshall-sycl, gaussian-sycl, pathfinder-sycl | FUND-1, IMPR-1, IMPR-2 | R6 (M1) |
| lud-sycl | FUND-1, IMPR-1 | excluded (BUG-5) |
| memcpy-sycl | IMPR-3 | R6 (M1) |
| babelstream-sycl | FUND-3 | excluded (BUG-5) |
| black-scholes-sycl, binomial-sycl, mandelbrot-sycl | IMPR-5, clspv math lowering | R6 (M1) |
| softmax-sycl | subgroups | R6 (M1) |
| scan-sycl, layernorm-sycl, shuffle-sycl | local mem, barriers, subgroups (sg 8 vs 64) | excluded (BUG-5) |
| histogram-sycl, atomicReduction-sycl | atomics lowering | R6 (M1) |
| atomicCost-sycl | atomics lowering | excluded (BUG-5) |
| su3-sycl | general codegen | R6, R8 (M1) |
| lavaMD-sycl | general codegen | excluded (BUG-5) |
| miniWeather-sycl | general codegen | not built (MPI) |

Suggested additions based on R6–R8: an integer-heavy index-decomposition
kernel to isolate IMPR-6; a memcpy sweep beyond 2 MB for IMPR-3;
ISA comparison for histogram (IMPR-10).

Machines that would add the most information: a discrete AMD GPU (non-APU
memory types), NVIDIA (proprietary Vulkan vs CUDA), Intel (ANV vs L0), and
a mobile/Adreno device.
