# Vulkan backend: performance opportunities

Every performance improvement opportunity found while analysing the
AdaptiveCpp Vulkan backend, with the evidence, a proposed change, the expected
gain and how to check it. The evidence itself lives in
[`../vulkan-perf/findings.md`](../vulkan-perf/findings.md) (finding IDs
`IMPR-n`, runs `Rn`); this file is the planning view of that material. What
users can do today is in
[`../fundamental-limitations/vulkan-performance-limitations.md`](../fundamental-limitations/vulkan-performance-limitations.md).

Measured with AdaptiveCpp `develop` at `f4ff3681` on two laptops:

* **M1**: AMD Ryzen 7 PRO 8840HS with Radeon 780M (RDNA3 APU). Vulkan (RADV)
  vs HIP; llvmpipe vs OpenMP.
* **M2**: Intel Core Ultra 7 155H with Intel Arc Graphics (Meteor Lake) and
  an NVIDIA RTX 500 Ada Generation Laptop GPU. Vulkan (ANV) vs OpenCL
  (Intel compute-runtime) on the iGPU, Vulkan (NVIDIA driver) vs CUDA on the
  dGPU, llvmpipe vs the Intel OpenCL CPU runtime and OpenMP.

Ratios are Vulkan time / native time on the same hardware (above 1 means
Vulkan is slower). The **gain evidence** column says how the expected gain is
known:

* **A/B**: the fix was simulated by changing the generated code or the
  clspv flags of an unchanged program (`tools/clspv_shim.c`,
  `tools/patch_barrier.sh`);
* **emulated**: a source-level change produces what the fix would produce
  (e.g. `unsigned` instead of `size_t` indices);
* **estimate**: derived from profiles, syscall counts or per-call costs;
  not demonstrated.

Effort is a rough guess (S/M/L), not an estimate by the code owners.

## Where the gaps are

Parity already exists for large bandwidth-bound kernels (64–256 MB:
NVIDIA 113 vs 111 GB/s, ANV 63 vs 64 GB/s; RADV 66–73 vs 66–70 GB/s) and for
long, simple kernels such as gaussian, the black-scholes kernel and
atomicReduction (within ±15% on all three GPUs). The gaps:

| Symptom (M2 unless noted) | Vulkan / native | Main causes |
|---|---|---|
| Any kernel that calls `sycl::fma` | 40–73× slower than `a*b + c` | [IMPR-16](#1-impr-16-hardware-fma-for-syclfma) |
| Small host↔device copies (memcpy-sycl, 64 B–64 KB) | NVIDIA 28–117×, ANV 6–16×; M1 RADV ~9–20× | [IMPR-3](#7-impr-3-staging-buffers-for-hostdevice-copies) |
| Large D2H copies on Intel | 0.16 vs 13 GB/s | [IMPR-13](#2-impr-13-choose-the-staging-memory-type-by-copy-direction) |
| mandelbrot "parallel time" (kernel + 8 MB D2H) | NVIDIA 4.2×, ANV 18×; M1 RADV 13× | IMPR-3, IMPR-13, IMPR-15 |
| black-scholes "processing time" (with transfers) | NVIDIA 2.0×, ANV 3.7×; M1 RADV 1.8× | IMPR-3, IMPR-13 |
| bfs on a 1999-level grid | timed kernels (2 launches + `wait()` per level) NVIDIA 2.05×, ANV 2.70×; process wall time (adds two 1-byte copies per level) 2.41× / 1.48× | [IMPR-1](#11-impr-1-batch-operations-into-fewer-submissions) (kernels), IMPR-3 (wall time) |
| floydwarshall (1024 dependent launches per run) | NVIDIA 1.27×; M1 RADV 1.37×; ANV 0.96× | IMPR-1, [IMPR-2](#6-impr-2-cheaper-command-buffer-recycling) |
| pathfinder | NVIDIA 1.66×, ANV 2.73× | [IMPR-14](#3-impr-14-barriers-that-only-fence-local-memory-when-asked) (nearly all of NVIDIA's gap, a little of ANV's) |
| binomial | NVIDIA 1.90×, ANV 1.88× | IMPR-14 (part) |
| nbody | NVIDIA 1.44×, ANV 2.48×; M1 RADV 1.23× | [IMPR-5](#4-impr-5-fma-contraction), [IMPR-11](#5-impr-11-rsqrt-to-the-hardware-instruction) (NVIDIA); FUND-2, [IMPR-6](#8-impr-6-32-bit-index-arithmetic) (ANV) |
| su3 (index decomposition with `size_t` div/mod) | M1 RADV 2.62×; NVIDIA 1.34×; ANV 1.02× | IMPR-6, IMPR-5 |
| mandelbrot kernel | NVIDIA 3.66×, ANV 3.48× | [IMPR-15](#10-impr-15-reuse-device-allocations) (ANV), IMPR-5, rest [unexplained](#open-investigations) |
| histogram atomics | NVIDIA 1.84× (local), ANV 1.1–3.3×; M1 RADV 1.42–1.44× | [IMPR-10](#12-impr-10-atomics), cause unknown |
| Serial loops, struct copies | `single_task` sum NVIDIA 3.5×; nbody update kernel 1.9× | [IMPR-12](#9-impr-12-runtime-unrolling-and-loadstore-vectorization) |

## Summary and suggested order

Quick wins with A/B or emulated evidence first, then larger changes, then
items that need investigation before any work.

| # | ID | Opportunity | Main beneficiaries | Measured gap | Gain evidence | Effort |
|---|---|---|---|---|---|---|
| 1 | IMPR-16 | Hardware FMA for `sycl::fma` via `SPV_KHR_fma` | `sycl::fma` on devices with `VK_KHR_shader_fma` (NVIDIA here) | FMA chain NVIDIA 499 ms vs CUDA 2.8 ms | A/B: 499 → 4.5 ms | S |
| 2 | IMPR-13 | Staging memory type chosen by copy direction | D2H copies on Intel (ANV) | D2H 0.16 vs 13 GB/s | emulated: host reads 0.16 → 15.2 GB/s | S |
| 3 | IMPR-14 | Local-only barriers when the code asks for them | kernels with barriers and local memory | pathfinder NVIDIA 1.66×, binomial 1.90× | A/B: −37% / −30% (NVIDIA), −10% / −11% (ANV) | S–M |
| 4 | IMPR-5 | FMA contraction (`--cl-mad-enable` when allowed) | FP-heavy kernels on NVIDIA and AMD | `x*a+b` chain NVIDIA 2.5×; nbody 1.44× | A/B: nbody accel −15% (NVIDIA), +11% GFLOP/s (RADV); none on ANV | S |
| 5 | IMPR-11 | `rsqrt` to `InverseSqrt` | `sycl::rsqrt` in hot loops | nbody accel NVIDIA 7.58 vs 5.42 ms | A/B: −14% (`native::rsqrt`), −23% (`--cl-native-math`) | S |
| 6 | IMPR-2 | One semaphore read per refresh; growable command-buffer pool | launch-heavy code | 22% of RADV submit time | estimate | S |
| 7 | IMPR-3 | Pooled, persistently mapped staging; direct path for host-visible memory | every host↔device copy | small copies NVIDIA ~40–80×, RADV ~4–20×, ANV ~2× | estimate: ~330 of ~570 µs per NVIDIA copy is allocate/map/free | M |
| 8 | IMPR-6 | 32-bit index arithmetic | index-heavy kernels; every kernel a little | `idx.cpp` NVIDIA 15×; su3 RADV 2.6× | emulated: `unsigned` indices 15× (NVIDIA), 3.9× (ANV) | M |
| 9 | IMPR-12 | Runtime unrolling and load/store vectorization before clspv | loops with runtime trip counts, struct copies | serial loop NVIDIA 3.5×; nbody update 1.9× | emulated: `#pragma unroll 8` 2.5–3.7× | M |
| 10 | IMPR-15 | Reuse device allocations | code that allocates per iteration | mandelbrot kernel on ANV 2× with a fresh buffer | emulated: 1.85–2.08 → 0.85–1.05 ms | M |
| 11 | IMPR-1 | Batch operations into fewer submissions | many small or dependent kernels | host cost per launch NVIDIA 17.6 vs 4.0 µs; M1 RADV 48 vs 7 µs | estimate | L |
| 12 | IMPR-10 | Atomics | histogram-style atomics | NVIDIA 1.84×, ANV up to 3.3× | none; cause unknown | ? |
| 13 | IMPR-4 | Persistently mapped uniform ring; cheaper descriptor slots | kernels with more than 256 B of arguments | not measured | none | M |
| 14 | IMPR-7 | `NonWritable` / `Restrict` decorations on kernel pointers | uniform loads (FUND-2) | — | speculative | S |
| 15 | IMPR-8 | `VkPipelineCache` | first launch per process | hidden by the drivers' shader caches | none | S |

---

## 1. IMPR-16: Hardware FMA for `sycl::fma`

[findings.md IMPR-16](../vulkan-perf/findings.md#impr-16)

**Problem.** SYCL requires `sycl::fma` to be correctly rounded. clspv
implements OpenCL `fma` for finite inputs as a software routine of ~200
integer instructions; the hardware instruction is only used for NaN/Inf
inputs. A chain of `sycl::fma` runs 40–73× slower than the same chain written
`x*a + b`, and users have no fast spelling: `sycl::mad` and `a*b + c` are an
unfused multiply and add ([IMPR-5](#4-impr-5-fma-contraction)).

**Where.**
* `__acpp_sscp_fma_f32` / `_f64` in `src/libkernel/sscp/clspv/math.cpp` call
  OpenCL `fma`.
* clspv options: fixed list plus `applyBuildOption` in
  `src/compiler/llvm-to-backend/clspv/LLVMToCLSPV.cpp`, fed from
  `vk_queue.cpp` through `setBuildOption` (as for `-max-pushconstant-size`).
* Device extensions: `optional_exts` in
  `src/runtime/vk/vk_hardware_manager.cpp`.
* Capability check: `verify_spv_capabilities` in
  `src/runtime/vk/vk_code_object.cpp`.
* clspv emits `OpFmaKHR` only if `--spv-khr-fma` lists every float width it
  has enabled, and fp16 and fp64 are enabled by default (`lib/Compiler.cpp`).

**Evidence (R18).** 1024-step dependent chain over 4M work-items:

| | `x*a + b` | `sycl::fma(x, a, b)` |
|---|---|---|
| Vulkan NVIDIA | 6.83 ms | **499 ms** |
| CUDA | 2.77 ms | 2.76 ms |
| Vulkan ANV | 8.51 ms | **351 ms** |
| OpenCL (Intel GPU) | 4.80 ms | 4.43 ms |
| llvmpipe | 533 ms | **5333 ms** |

`-ffast-math` changes nothing. `vulkaninfo`: NVIDIA exposes
`VK_KHR_shader_fma` with `shaderFmaFloat16/32/64`; ANV and llvmpipe (Mesa
26.0.8) do not.

**Proposal.**
1. When the device exposes `VK_KHR_shader_fma`, enable it and chain
   `VkPhysicalDeviceShaderFmaFeaturesKHR` into device creation.
2. Pass `--spv-khr-fma=<widths>` to clspv as a build option, covering every
   width clspv has enabled (or disable the widths the device lacks). Binaries
   built with it must not be reused on devices without the feature; check how
   the kernel cache keys the existing per-device options.
3. Accept capability `FMAKHR` (6030) in `verify_spv_capabilities`.
4. Without the extension, keep the software routine: the Vulkan spec only
   guarantees `GLSL.std.450 Fma` the precision of "OpFMul followed by OpFAdd",
   while `OpFmaKHR` is "Correctly rounded". A relaxed mode (e.g. under
   `-ffast-math`) could use `GLSL.std.450 Fma` instead.

**Expected gain.** A/B (`CLSPV_EXTRA=--spv-khr-fma=16,32,64` through
`clspv_shim.c`): the NVIDIA chain drops from 499 to 4.49 ms, below
`x*a + b` (6.8 ms; CUDA 2.8 ms). Nothing on devices without the extension
unless the relaxed mode is used.

**Validate with.** `tools/uguide.cpp`; `spirv-dis` of the JIT cache (one
`OpFmaKHR`, no integer sequence); the validation layer (no capability or
feature errors); HeCBench kernels that call `sycl::fma`.

**Risks.** The `--spv-khr-fma` widths must match what clspv has enabled,
otherwise clspv silently keeps the routine for all widths (as with
`--spv-khr-fma=32` alone, which only changed the NaN/Inf path). Results do
not change: both paths are correctly rounded.

## 2. IMPR-13: Choose the staging memory type by copy direction

[findings.md IMPR-13](../vulkan-perf/findings.md#impr-13)

**Problem.** Staging buffers prefer `HOST_VISIBLE|HOST_COHERENT|HOST_CACHED`
and fall back to `HOST_VISIBLE|HOST_COHERENT`. On ANV (Meteor Lake) the
coherent types are uncached (write-combined) and the cached types are not
coherent, so staging lands in write-combined memory, and the host `memcpy`
out of it for device-to-host copies reads at 0.16 GB/s.

**Where.** `vk_allocator::staging_allocate` (type selection);
`cleanup_staging_buffers` (the host `memcpy` for D2H).

**Evidence.** R11 `xfer.cpp`: D2H 0.16 GB/s from 16 MB up (OpenCL 11–13 GB/s);
1 MB in 4.6 ms (OpenCL 0.24 ms); 256 MB in 1.66 s (OpenCL 21 ms).
`memtype.cpp`: host reads 0.16 GB/s from the uncached coherent type and
15.2 GB/s from the cached non-coherent type; host writes 20.3 and 14.8 GB/s.
R14: memcpy-sycl D2H 5.3 ms at 1 MB and 9.6 ms at 2 MB (OpenCL ~0.1 ms);
mandelbrot "parallel time" 45 vs 2.45 ms.

**Proposal.** For D2H, prefer a `HOST_CACHED` type even when it is not
`HOST_COHERENT`, and call `vkInvalidateMappedMemoryRanges` after the GPU
copy and before the host `memcpy`. For H2D, write-combined coherent memory is
fine (20 GB/s host writes). Apply the same rule to any direct mapping of
device allocations ([IMPR-3](#7-impr-3-staging-buffers-for-hostdevice-copies)
item 2): NVIDIA's ReBAR device memory reads at 0.02 GB/s.

**Expected gain.** Emulated by `memtype.cpp`: the host read goes from 0.16 to
15.2 GB/s, so large D2H copies on ANV should approach the H2D rate
(3.3–3.6 GB/s today, ~20×), after which IMPR-3 applies.

**Validate with.** `xfer.cpp` (D2H GB/s), `memtype.cpp`, memcpy-sycl, the
mandelbrot "parallel time" and black-scholes "processing time".

**Risks.** Invalidate ranges must respect `nonCoherentAtomSize`. Choose by
memory property flags rather than by vendor; other drivers expose other type
sets.

## 3. IMPR-14: Barriers that only fence local memory when asked

[findings.md IMPR-14](../vulkan-perf/findings.md#impr-14)

**Problem.** Every work-group barrier also orders global memory.
`sscp_barrier()` maps `fence_space::local_space` and `global_and_local` to the
same builtin, which the clspv libkernel implements as
`barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)`; clspv emits
`OpControlBarrier Workgroup Workgroup 0x148`
(AcquireRelease|UniformMemory|WorkgroupMemory). ANV adds a global (UGM) fence
before each such barrier; IGC (OpenCL) only fences shared local memory.

**Where.** `include/hipSYCL/sycl/libkernel/detail/device_barrier.hpp`
(`sscp_barrier`), the SSCP builtin `__acpp_sscp_work_group_barrier` (shared
by all SSCP backends), `src/libkernel/sscp/clspv/barrier.cpp`.

**Evidence (R16, `patch_barrier.sh` rewriting `0x148` to `0x108` in the
cached SPIR-V).**

| | Vulkan ANV | Vulkan NVIDIA | Native |
|---|---|---|---|
| pathfinder, total kernel time | 3.38 → 3.03 s | 3.45 → 2.18 s | OpenCL 1.22 s, CUDA 2.08 s |
| binomial, avg kernel | 7.67 → 6.80 ms | 10.09 → 7.06 ms | OpenCL 4.01 ms, CUDA 5.23 ms |
| histogram | unchanged | unchanged | |

binomial still passes its check with the patched barriers.

**Proposal.** Pass the fence space (e.g. an address-space mask) through the
SSCP barrier builtin, so that `local_space` barriers become
`barrier(CLK_LOCAL_MEM_FENCE)`. Keep global semantics for `global_space`,
`global_and_local` and `group_barrier`.

**Expected gain.** A/B: pathfinder −37% on NVIDIA (within 5% of CUDA) and
−10% on ANV; binomial −30% on NVIDIA and −11% on ANV. Only kernels that ask
for local-only fences benefit (e.g. `nd_item::barrier(access::fence_space::local_space)`).

**Validate with.** `patch_barrier.sh` for further A/B checks; after the
change: pathfinder, binomial, histogram, and the SPIR-V barrier semantics.

**Risks.** The builtin's signature changes for every SSCP backend. Code that
relies on a local barrier to order global memory would break, as it would on
other SYCL implementations.

## 4. IMPR-5: FMA contraction

[findings.md IMPR-5](../vulkan-perf/findings.md#impr-5)

**Problem.** clspv is invoked without `--cl-mad-enable`, so it decorates every
float operation `NoContraction` although the input IR carries `contract`
flags: `a*b + c` is never fused. Users cannot work around it today:
`-ffast-math` changes nothing, `sycl::mad` produces the same unfused
`NoContraction` code, and `sycl::fma` is a software routine (IMPR-16).

**Where.** The fixed clspv argument list in `LLVMToCLSPV.cpp`; clspv's
handling of `contract`.

**Evidence.** M1: HeCBench nbody RADV 690 GFLOP/s vs HIP 850; 765 with
`--cl-mad-enable` (JIT cache swapped by hand). su3 (FP64): 12 `v_mul_f64` +
10 `v_add_f64` where HIP has 6 `v_fma_f64`. M2 (R16, `clspv_shim.c`): NVIDIA
nbody accel 7.72 → 6.55 ms (CUDA 5.42), mandelbrot kernel 1.15 → 1.04 ms
(CUDA 0.71); ANV and llvmpipe unchanged. R18: 1024-step `x*a+b` chain NVIDIA
6.83 ms vs CUDA 2.77 ms.

**Proposal.** Pass `--cl-mad-enable` when the source flags allow contraction,
or better, make clspv honour per-instruction `contract` flags. Keep a way to
opt out for bit-reproducibility (e.g. `-ffp-contract=off`).

**Expected gain.** A/B: nbody accel −15% and mandelbrot kernel −10% on NVIDIA;
+11% GFLOP/s for nbody on RADV; no measurable change on ANV or llvmpipe.

**Validate with.** `prof.cpp`, `uguide.cpp` (`x*a+b` chain), `mb.cpp` (nbody
accel), HeCBench nbody, mandelbrot, su3; SPIR-V without `NoContraction` on
fusable operations.

**Risks.** Results change in the last bits, as they do on the native backends,
which contract by default.

## 5. IMPR-11: `rsqrt` to the hardware instruction

[findings.md IMPR-11](../vulkan-perf/findings.md#impr-11)

**Problem.** `sycl::rsqrt` maps to OpenCL `rsqrt`, which clspv implements as
`Sqrt`, range scaling, `OpFDiv 1.0, x` and selects instead of
`GLSL.std.450 InverseSqrt`. Only `sycl::native::rsqrt` reaches the hardware
instruction. The CUDA backend turns all three spellings (`rsqrt`,
`native::rsqrt`, `1/sqrt`) into `rsqrt.approx.f32`, and IGC turns OpenCL
`rsqrt` into `math.rsqt`, so part of the gap is a difference in precision
policy.

**Where.** `__acpp_sscp_rsqrt_f32` in `src/libkernel/sscp/clspv/math.cpp`.

**Evidence.** R16 `rsq.cpp` (nbody accel kernel; `sycl::rsqrt` /
`native::rsqrt` / `1/sqrt`): NVIDIA 7.58 / 6.54 / 7.57 ms (CUDA 5.42 for all
three); ANV 9.3–10.1 / 7.9–8.5 / 9.1–9.2 ms (OpenCL 3.0 / 2.77 / 2.26 ms);
llvmpipe 68 / 54–59 / 67 ms. With `--cl-native-math`: NVIDIA 7.72 → 5.95 ms,
ANV ~10 → ~7.7 ms. R18: division is *not* a gap; a division chain runs faster
on Vulkan than natively (NVIDIA 4.13 vs CUDA 5.13 ms; ANV 3.49 vs OpenCL
4.01 ms).

**Proposal.** Lower `rsqrt` to `InverseSqrt`: Vulkan's 2 ULP requirement for
it matches OpenCL's `rsqrt`; the difference is denormal and edge-case
handling. Alternatively, pass `--cl-native-math` / `--cl-fast-relaxed-math` /
`--cl-denorms-are-zero` under `-ffast-math`.

**Expected gain.** A/B: nbody accel −14% on NVIDIA and −15% on ANV with the
`native::rsqrt` code; −23% on both with `--cl-native-math`.

**Validate with.** `rsq.cpp`, `mb.cpp` (nbody accel), HeCBench nbody;
precision tests around denormals and infinities.

**Risks.** Denormal and edge-case results differ from OpenCL `rsqrt`.

## 6. IMPR-2: Cheaper command-buffer recycling

[findings.md IMPR-2](../vulkan-perf/findings.md#impr-2)

**Problem.** The refresh loop in `vk_queue::get_command_buffer` reads the
timeline semaphore (`getCounterValue`, an ioctl on RADV) once per in-flight
command buffer. The pool is fixed at 24 command buffers
(`_cmd_buf_alloc_size`), so the host blocks once 24 operations are in flight.

**Where.** `src/runtime/vk/vk_queue.cpp` (`get_command_buffer`),
`include/hipSYCL/runtime/vk/vk_queue.hpp` (`_cmd_buf_alloc_size`).

**Evidence.** M1 sampling (R4): 22% of RADV submit time in
`get_command_buffer` → `drmSyncobjQuery2`; on llvmpipe the launch loop spends
72% blocked in `get_command_buffer` waiting for the pool to drain. Not
measured on M2.

**Proposal.** Read the counter once per refresh; allocate more command
buffers on demand instead of blocking (with a cap).

**Expected gain.** Estimate: up to ~22% of the host submit cost on RADV; no
host stalls with more than 24 operations in flight.

**Validate with.** `mb.cpp` (pipelined launch cost), `lat.cpp` with
`strace -f -c` (ioctls per launch), `samp.sh`.

**Risks.** Low; cap the pool so a slow device cannot make it grow without
bound.

## 7. IMPR-3: Staging buffers for host↔device copies

[findings.md IMPR-3](../vulkan-perf/findings.md#impr-3)

**Problem.** Every copy to or from host memory allocates a new staging
`VkBuffer` + `VkDeviceMemory`, records a GPU copy, and hands over to the
`_host_worker` thread, which waits on the semaphore, maps, `memcpy`s, unmaps
and signals the user thread. Device USM cannot be host memory
([FUND-4](../vulkan-perf/findings.md#fund-4)), so every host copy pays this.

**Where.** `vk_allocator::staging_allocate`;
`setup_staging_buffers` / `cleanup_staging_buffers` and `_host_worker` in
`src/runtime/vk/vk_queue.cpp`.

**Evidence.**

| | Vulkan | Native |
|---|---|---|
| M1 RADV, 4 B D2H + wait | 151 µs | HIP 34 µs |
| M1 RADV, memcpy-sycl 64 B–64 KB | 117–159 µs | HIP 5.5–20 µs |
| M2 NVIDIA, 16 B–4 KB + wait | 565–738 µs | CUDA 9–15 µs |
| M2 NVIDIA, 16–256 MB | 2.1–3.0 GB/s | CUDA 8.1–8.4 GB/s |
| M2 ANV, 16 B–4 KB + wait | 179–235 µs | OpenCL 99–107 µs |
| M2 ANV, H2D 16–256 MB | 3.3–3.6 GB/s | OpenCL 11–13 GB/s |
| M2 llvmpipe, ≥ 1 MB | ~4 GB/s | OpenCL CPU 25–67 GB/s |

Per copy: ~11 ioctl + mmap/munmap on RADV; 10 ioctl + `openat`/`close` +
mmap/munmap on NVIDIA. On NVIDIA, allocating, mapping and freeing a small
staging buffer of the type used costs 334 µs (`memtype.cpp`), and sampling
shows `vkAllocateMemory` on the submitting thread and `vkFreeMemory` on the
worker. End to end (R14): memcpy-sycl NVIDIA 7–117× CUDA and ANV 6–91×
OpenCL; black-scholes "processing time" 2.0× / 3.7×; mandelbrot "parallel
time" 4.2× / 18×.

**Proposal.**
1. Pool staging buffers and keep them persistently mapped. This is the part
   that matters most on NVIDIA.
2. If the user allocation is `HOST_VISIBLE` (every allocation on APUs, ReBAR
   on dGPUs), skip the GPU copy: wait for prior work and `memcpy` through a
   persistent mapping. Only for H2D or `HOST_CACHED` memory: host reads run at
   0.02 GB/s from NVIDIA ReBAR and 0.16 GB/s from ANV device memory
   ([IMPR-13](#2-impr-13-choose-the-staging-memory-type-by-copy-direction)).
3. Persistently map device allocations instead of mapping per use.
4. Not evaluated: split large copies into chunks so the GPU copy and the host
   `memcpy` overlap.

**Expected gain.** Estimate: removing allocation, mapping and freeing takes
~330 µs off each ~570 µs NVIDIA copy; the worker-thread hand-off remains.
Item 2 would make small H2D copies on APUs and ReBAR devices close to a
`memcpy`.

**Validate with.** `xfer.cpp` (latency and bandwidth sweep), `memtype.cpp`,
`lat.cpp` with `strace -f -c` and `samp.sh`, memcpy-sycl, black-scholes, the
mandelbrot "parallel time", bfs on the grid graph.

**Risks.** Memory held by the pool; thread safety between the submitting
thread and the host worker; non-coherent types need flushes and invalidates.

## 8. IMPR-6: 32-bit index arithmetic

[findings.md IMPR-6](../vulkan-perf/findings.md#impr-6)

**Problem.** Index arithmetic in the clspv path is 64-bit, including division
and modulo by constants, which reach the driver as SPIR-V `OpUDiv %ulong` /
`OpUMod %ulong`. The PTX path narrows the same `size_t` arithmetic to 32 bits:
its ids start as i32 values, and the `GlobalSizesFitInI32OptPass`
assumptions let InstCombine narrow. In the clspv flavour `get_group_id` /
`get_local_id` are opaque i64 calls (`src/libkernel/sscp/clspv/core.cpp`), so
there is no `zext` to narrow. In addition, LLVM widens `int` loop counters
that index memory to i64 (the `spir64` data layout declares i64 native) and
nothing strength-reduces the address computation; Intel Xe-LPG has no 64-bit
integer ALU, so ANV emulates each step.

**Where.** `src/libkernel/sscp/clspv/core.cpp` (id builtins), the clspv
flavour's data layout and pass pipeline (`LLVMToCLSPV.cpp`).

**Evidence.** M1 su3: RADV 12.3 vs HIP 32.3 GFLOP/s, ISA dominated by an
expanded 64-bit division (125 `v_cmp_le_u64`, 334 `v_cndmask_b32`, ...) where
HIP uses five `v_mad_u64_u32`. M2 `idx.cpp` (su3-style decomposition,
`size_t` vs `unsigned`): NVIDIA 246.9 vs 16.4 ms (CUDA 17.5 ms for both), ANV
49.8 vs 12.9 ms (OpenCL 19.8 vs 12.2), llvmpipe 291 vs 168 ms. ANV nbody inner
loop: 64-bit loop counter and emulated 64-bit `mul`/`mach`/carry sequences,
~45 integer for ~20 float instructions. su3: NVIDIA 1.34× CUDA, ANV at parity
(memory-bound there).

**Proposal.**
1. Make the clspv id builtins return i32 values that are zero-extended, as
   the PTX builtins do (Vulkan's `WorkgroupId` / `LocalInvocationId` are
   32-bit anyway), so the existing range assumptions narrow the arithmetic.
2. Keep loop counters 32-bit: a data layout without native i64
   (`n8:16:32`), or loop strength reduction before clspv.
3. Lower any remaining `udiv`/`urem` by a constant to a multiply-high
   sequence in IR.

**Expected gain.** Emulated (`unsigned` indices): 15× on NVIDIA, 3.9× on ANV
and 1.7× on llvmpipe for `idx.cpp`; up to 2.6× for su3 on RADV (estimate).
Small but non-zero in every kernel.

**Validate with.** `idx.cpp`, su3, nbody on ANV (`INTEL_DEBUG=cs` instruction
counts), `ACPP_S2_DUMP_IR_FINAL=1` (no `udiv i64`), SPIR-V (no
`OpUDiv %ulong`).

**Risks.** Only valid when global sizes fit in 32 bits, the same condition the
PTX path already relies on. Pointer arithmetic itself stays 64-bit
(physical storage buffer addresses).

## 9. IMPR-12: Runtime unrolling and load/store vectorization

[findings.md IMPR-12](../vulkan-perf/findings.md#impr-12)

**Problem.** AdaptiveCpp's O3 pipeline runs without a target machine for
every flavour. The PTX flavour then runs `opt --mcpu=sm_XX -O3` (NVPTX TTI
enables runtime unrolling) and `llc -O3` (which runs the load/store
vectorizer); the clspv flavour hands the target-independent IR to clspv, and
the Vulkan driver compilers do not runtime-unroll loops with unknown trip
counts. Adjacent scalar accesses are not merged either.

**Where.** `src/compiler/llvm-to-backend/clspv/LLVMToCLSPV.cpp`, compared with
`src/compiler/llvm-to-backend/ptx/LLVMToPtx.cpp`.

**Evidence.** The `single_task` sum in `mb.cpp` reaches CUDA unrolled ×8 with
`ld.global.v4.b32` / `v2` loads, and reaches the Vulkan driver as a loop with
one scalar `OpLoad` per iteration and a 64-bit induction variable: NVIDIA
805 vs 226 µs (R9). nbody's 40-byte particle struct is copied with 10 scalar
loads and 10 scalar stores in SPIR-V, 5 `ld.global.v2` and 4 `st.global.v2`
in PTX; update-kernel device time 37 vs 19.5 µs on NVIDIA. R18, serial sum of
65536 floats with `#pragma unroll 8` as a stand-in for the fix: NVIDIA
3.15 → 1.14 ms (CUDA 0.91 ms without the pragma), ANV 6.41 → 2.56 ms (OpenCL
4.72 → 1.98 ms), llvmpipe 1.05 → 0.28 ms.

**Proposal.** Before handing IR to clspv, run `LoopUnrollPass` with runtime
unrolling and `LoadStoreVectorizerPass` (the equivalent of the PTX flavour's
target-aware `opt`/`llc` step). Check that clspv and the drivers accept the
resulting vector accesses on physical storage buffer pointers.

**Expected gain.** Emulated: 2.5–3.7× on latency-bound serial loops on all
three Vulkan drivers. Unknown for struct copies.

**Validate with.** `uguide.cpp` (serial sum), `mb.cpp` (`single_task`, update
kernel), SPIR-V inspection (vector `OpLoad`s), HeCBench nbody.

**Risks.** Code size and register pressure; unroll factors may need tuning
per device (the same pragma made the CUDA build 2.7× slower).

## 10. IMPR-15: Reuse device allocations

[findings.md IMPR-15](../vulkan-perf/findings.md#impr-15)

**Problem.** Every `malloc_device` creates a new `VkDeviceMemory` and `free`
releases it. On ANV the first kernel that touches a fresh allocation is
slow; on NVIDIA the allocation itself is expensive. The OpenCL backend on the
same Intel GPU shows no first-use penalty.

**Where.** `vk_allocator::raw_allocate` / `raw_free`
(`src/runtime/vk/vk_allocator.cpp`).

**Evidence.** R16 `mandel.cpp` (kernel + `wait()`): ANV 1.85–2.08 ms with a
fresh 8 MB buffer per launch vs 0.85–1.05 ms reused (OpenCL 0.67–0.73 ms
either way); NVIDIA unaffected (1.10–1.28 ms either way). Allocating, mapping,
unmapping and freeing 4 KB of NVIDIA ReBAR memory costs 119 µs (R11). Part of
HeCBench mandelbrot's 3.48× on ANV, which allocates its output in every
iteration.

**Proposal.** Keep freed allocations in a size-bucketed cache, or sub-allocate
from larger blocks, instead of calling `vkAllocateMemory` / `vkFreeMemory` per
`malloc_device` / `free`.

**Expected gain.** Emulated: 2× for the mandelbrot kernel on ANV;
allocation-heavy code on NVIDIA saves ~0.1 ms per allocation (estimate).

**Validate with.** `mandel.cpp`, HeCBench mandelbrot, `memtype.cpp`
(allocation cost).

**Risks.** Memory stays reserved after `free` (trim under pressure).
Sub-allocation needs alignment and correct device-address offsets, and the
allocation lookup used by copies (`find_user_alloc`, see BUG-4) must keep
working.

## 11. IMPR-1: Batch operations into fewer submissions

[findings.md IMPR-1](../vulkan-perf/findings.md#impr-1) ·
[FUND-1](../vulkan-perf/findings.md#fund-1)

**Problem.** Each kernel launch, copy or fill gets its own command buffer and
its own `vkQueueSubmit`, waiting on the previous timeline value with
`eAllCommands`.

**Where.** `vk_queue::submit_sscp_kernel_from_code_object`, `submit_memcpy`,
`submit_memset`, ... calling `begin_command_buffer` /
`submit_command_buffer` (`src/runtime/vk/vk_queue.cpp`).

**Evidence.** Host cost per pipelined empty-kernel launch: M1 RADV 48 µs vs
HIP 7 µs (56% of it in `vkQueueSubmit`), llvmpipe 35 µs with ~57 futex calls;
M2 NVIDIA 17.6 vs CUDA 4.0 µs (no syscalls: user-space cost), llvmpipe ~79
futex calls per launch. Per dependent kernel at 1–4 MB: NVIDIA +9–11 µs
(R10). End to end: floydwarshall NVIDIA 1.27×, M1 RADV 1.37×; bfs on the
1999-level grid NVIDIA 2.05×, ANV 2.70× (kernel time). On the Intel GPU the
native OpenCL stack is not faster (pipelined 41.6 vs 33.7 µs end to end,
floydwarshall 0.96×), so the gain there may be small.

**Proposal.** Record consecutive in-order operations into one command buffer
separated by `vkCmdPipelineBarrier` (compute→compute, transfer↔compute).
Flush on `wait()`, host tasks, cross-queue dependencies, staging operations,
and a batch-size or time threshold.

**Expected gain.** Estimate: host cost per back-to-back kernel closer to the
native ~4–7 µs on NVIDIA and RADV. Less where the host waits after every
few operations (bfs: two kernels per `wait()`), because each host round trip
remains
([FUND-1](../fundamental-limitations/vulkan-performance-limitations.md#1-every-submission-and-every-synchronisation-has-a-fixed-cost)).

**Validate with.** `mb.cpp` (pipelined and `wait()` launch costs), `bw.cpp`
(per dependent kernel), `lat.cpp` with `strace -f -c`, floydwarshall,
gaussian, bfs (grid).

**Risks.** Delayed submission adds latency unless flushes are well placed;
barriers between transfer and compute work must be correct; per-operation
profiling timestamps in a shared command buffer (and BUG-2's query pool
lifetime); llvmpipe's device-side cost per submit.

## 12. IMPR-10: Atomics

[findings.md IMPR-10](../vulkan-perf/findings.md#impr-10)

**Problem.** Histogram kernels with local- and global-memory atomics are
slower than natively, while atomicReduction (one global atomic per work-group
after a reduction) is at parity. Cause unknown.

**Evidence.** M1 RADV: local atomics 429 vs 297 µs, global 673 vs 473 µs
(1.42–1.44×). M2 NVIDIA: local 134 vs 73 µs (1.84×), global 376 vs 338 µs
(1.11×); ANV over two passes: local 1.1–1.44×, global 2.2–3.3× (910–1363 vs
412 µs). Not the barrier semantics of IMPR-14 (R16).

**Next step.** Compare the ISA of the histogram kernels (`INTEL_DEBUG=cs` vs
the IGC dump; `RADV_DEBUG=shaders` vs HIP), and check the memory semantics and
scopes emitted by the clspv atomic lowering
(`src/libkernel/sscp/clspv/atomic.cpp`): overly strong scopes, or CAS loops
instead of native atomic adds.

## 13. IMPR-4: Per-launch uniform and descriptor overhead

[findings.md IMPR-4](../vulkan-perf/findings.md#impr-4)

**Problem.** `vk_kernel_pipeline::set_args` maps and unmaps uniform memory
per argument and calls `updateDescriptorSets` per launch;
`create_kernel_descriptors` calls `waitSemaphores` per slot to find a free
one, and there are only `MAX_INSTANCES = 10` slots per kernel
(`vk_code_object.hpp/.cpp`).

**Evidence.** Code inspection only. The kernels measured so far pass their
arguments as push constants (up to 256 B), so this path was never exercised.

**Proposal.** A persistently mapped ring buffer for uniforms; track slot
completion with one counter read; more (or growing) slots.

**Validate with.** A launch-latency tool with more than 256 B of kernel
arguments (not written yet).

## 14. IMPR-7: `NonWritable` / `Restrict` decorations

[findings.md IMPR-7](../vulkan-perf/findings.md#impr-7) ·
[FUND-2](../vulkan-perf/findings.md#fund-2)

**Idea.** Emit SPIR-V `NonWritable` / `RestrictPointer` for `const` /
`__restrict` kernel pointers, which might let RADV and ANV use scalar loads
for uniform addresses (FUND-2). Speculative: check first what ACO and ANV do
with these decorations on physical storage buffer pointers
(`RADV_DEBUG=shaders`: `s_load`; `INTEL_DEBUG=cs`: single-lane loads).

## 15. IMPR-8: `VkPipelineCache`

[findings.md IMPR-8](../vulkan-perf/findings.md#impr-8)

Pipelines are created with a null cache in every process. The drivers' own
on-disk shader caches currently hide most of the cost. Low priority; measure
the first launch per process with the driver cache disabled (e.g.
`MESA_SHADER_CACHE_DISABLE=true`) before doing anything.

---

## Related opportunities outside the Vulkan backend

| Where | Opportunity | Evidence |
|---|---|---|
| AdaptiveCpp OpenMP backend ([IMPR-9](../vulkan-perf/findings.md#impr-9)) | `nd_range` kernels are slow: nbody accel OpenMP 71.8 ms vs llvmpipe 61.9 ms and Intel OpenCL CPU 23.2 ms (M2; M1: 117 vs 46 ms); 41 µs per empty kernel on M2's 22-thread hybrid CPU | R1, R2, R9, R14 |
| AdaptiveCpp Vulkan runtime ([BUG-4](../bug-reports/BUG-4-vk-find-user-alloc-linear-scan.md)) | `find_user_alloc` scans all allocations for interior pointers on every copy | code inspection |
| clspv | Honour per-instruction `contract` flags instead of the global `--cl-mad-enable` (IMPR-5) | R5, R16 |
| clspv | Select the native `fma` per float width: with fp64 enabled, `--spv-khr-fma=32` does not use `OpFmaKHR` for fp32 (IMPR-16) | R18 |
| Mesa lavapipe ([FUND-3](../vulkan-perf/findings.md#fund-3)) | Physical storage buffer loads and stores become per-lane gathers and scatters (two `vpgatherqd`, eight scalar stores for `b[i] = 2*a[i] + 1`); llvmpipe streams at 0.5–0.7× the OpenMP bandwidth | R3, R10, R16 |
| Mesa RADV / ANV compilers ([FUND-2](../vulkan-perf/findings.md#fund-2)) | Scalar (SMEM / single-lane) loads for uniform physical storage buffer loads | R5, R16 |
| ANV sub-group size ([BUG-10](../bug-reports/BUG-10-vk-anv-subgroup-size.md)) | Kernels are compiled SIMD16 unless a size is required; IGC uses SIMD32 for nbody. Re-measure the ANV numbers once the fix (SIMD32) is merged: it may change performance as well as correctness | R15, R16 |

## Open investigations

Gaps that no finding explains yet:

| Gap | Numbers | Known so far | Next step |
|---|---|---|---|
| mandelbrot loop on NVIDIA | 1.04 ms with `--cl-mad-enable` vs CUDA 0.71 ms | the CUDA loop is not unrolled and differs from the SPIR-V mainly by its two FMAs (R16) | compare driver statistics or ISA (e.g. `VK_KHR_pipeline_executable_properties`) |
| pathfinder on ANV after IMPR-14 | 3.03 s vs OpenCL 1.22 s | the global fence is gone; still 2.5× | ISA comparison: `INTEL_DEBUG=cs` vs the IGC dump (SIMD width, SLM accesses) |
| binomial after IMPR-14 | NVIDIA 7.06 vs CUDA 5.23 ms; ANV 6.80 vs OpenCL 4.01 ms | — | ISA comparison |
| nbody on ANV | ~7.7 ms with `--cl-native-math` vs OpenCL ~3 ms | per-lane loads for uniform addresses (FUND-2), emulated 64-bit addressing (IMPR-6), SIMD16 vs IGC's SIMD32 | separate the three: SIMD32 via the BUG-10 fix, 32-bit loop counters via IMPR-6 |
| nbody update kernel on NVIDIA | 60–225 µs from run to run vs CUDA 21–28 µs; device time 37 vs 19.5 µs | the struct copy (IMPR-12) explains part of the device time | find the source of the variance (submission and wake-up path) |
| bfs on a random graph | kernel time NVIDIA 1.89×, ANV 1.27–1.41× | few levels, so per-level costs matter less than on the grid | not analysed |
| su3 on NVIDIA | 22.9 vs 30.8 GFLOP/s (1.34×) | on RADV: 64-bit division and no FP64 FMA (IMPR-6, IMPR-5) | check the NVIDIA SPIR-V for the same patterns |
| llvmpipe vs the CPU runtimes ([FUND-5](../vulkan-perf/findings.md#fund-5)) | 1.4–11× slower than the Intel OpenCL CPU runtime on most kernels | gathers for every buffer access (FUND-3), ~35 µs per launch | separate per-launch from per-work-item cost |
| `single_task` loop on M1 | RADV 1.68 vs HIP 1.17 ms | attributed to scalar loads (FUND-2); on NVIDIA the gap is unrolling (IMPR-12) | check whether HIP's loop is unrolled |

## Validating a change

Follow the measurement rules in
[findings.md](../vulkan-perf/findings.md#measurement-rules-learned-the-hard-way): discard the
first run after a build (JIT compilation), use a private `ACPP_APPDB_DIR` for
A/B experiments, time with wall clocks rather than SYCL event profiling
(BUG-1, BUG-2), run laptops on AC with the `performance` power profile,
interleave A/B runs on integrated GPUs, and only record HeCBench numbers whose
results were verified. Then add a run to findings.md and update the
finding's evidence and status.

| Tool (`../vulkan-perf/tools/`) | Use it for |
|---|---|
| `mb.cpp` | launch and synchronisation costs (IMPR-1, IMPR-2), nbody accel (IMPR-5, IMPR-11), `single_task` and update kernel (IMPR-12) |
| `bw.cpp` | cost per dependent kernel (IMPR-1), streaming bandwidth (FUND-3) |
| `lat.cpp`, `samp.sh`, `strace -f -c` | syscalls and host hot spots per launch or copy (IMPR-1, IMPR-2, IMPR-3) |
| `xfer.cpp`, `memtype.cpp` | copy latency and bandwidth, memory-type behaviour (IMPR-3, IMPR-13, IMPR-15) |
| `idx.cpp` | 64-bit index arithmetic (IMPR-6) |
| `prof.cpp` | contraction (IMPR-5) and profiling timestamps (BUG-1) |
| `rsq.cpp` | `rsqrt` lowering (IMPR-11) |
| `mandel.cpp` | allocation reuse (IMPR-15) |
| `uguide.cpp` | `fma`, division, unrolling (IMPR-5, IMPR-11, IMPR-12, IMPR-16) |
| `clspv_shim.c` + `clspv_wrapper.sh` | A/B of clspv flags without rebuilding (IMPR-5, IMPR-11, IMPR-16) |
| `patch_barrier.sh` | A/B of barrier semantics (IMPR-14) |
| `hecbench_build.sh`, `hecbench_suite.sh`, `hecbench_parse.py` | the end-to-end benchmark set of R14 |
