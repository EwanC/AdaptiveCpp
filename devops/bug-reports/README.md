# Bug reports

Upstream issue drafts for the bugs found while analysing the performance of
the Vulkan backend. IDs match section C of
[`../vulkan-perf/findings.md`](../vulkan-perf/findings.md), which holds the
full evidence. Each file contains:

* a header table: where to file, a suggested title, the affected backend and
  devices, and the evidence in findings.md;
* the issue body, between two horizontal rules, laid out like the
  repository's bug report template (`.github/ISSUE_TEMPLATE/bug_report.md`):
  summary, reproducer with its verbatim output, expected behaviour, setup,
  diagnostics and additional context.

Every reproducer was run on machine M2 of findings.md (Intel Core Ultra 7 155H
with Intel Arc Graphics, NVIDIA RTX 500 Ada Generation Laptop GPU) against
AdaptiveCpp `develop` at `f4ff3681`, and every code block is the program that
produced the quoted output. Some reports add observations from machine M1
(Radeon 780M).

## Index

| ID | Report | Area | Effect | Filing status |
|---|---|---|---|---|
| BUG-1 | [Event profiling ignores `timestampPeriod`](BUG-1-vk-profiling-timestamp-period.md) | Vulkan runtime | profiled durations 34–44× too short on Intel (ANV), 10× on AMD (RADV) | ready |
| BUG-2 | [Query pools destroyed while in use](BUG-2-vk-query-pool-lifetime.md) | Vulkan runtime | validation error on every driver; crash or hang on llvmpipe, device lost on RADV | ready |
| BUG-3 | [Invalid `ACPP_VISIBILITY_MASK` entries](BUG-3-visibility-mask-invalid-entries.md) | runtime, all backends | a misspelt backend makes every backend visible; an unmatched device filter is silent | ready; low priority (usability) |
| BUG-4 | [`find_user_alloc` linear scan](BUG-4-vk-find-user-alloc-linear-scan.md) | Vulkan runtime | O(number of allocations) per copy; inconsistent interval bounds, no wrong result seen | low priority; better as a small PR |
| BUG-6 | [Unsupported SPIR-V capabilities not rejected](BUG-6-vk-unsupported-spirv-capabilities.md) | Vulkan runtime | FP64 atomic add: `-nan` on ANV, process abort on llvmpipe | ready |
| BUG-7 | [Failed JIT compilation retried on every launch](BUG-7-sscp-failed-jit-retried-every-launch.md) | SSCP kernel cache (all JIT backends) | application appears to hang (~1.7 s per launch) | ready; file with BUG-11 |
| BUG-8 | [HeCBench lavaMD wrong results](BUG-8-vk-lavamd-wrong-results.md) | Vulkan | wrong results on all four Vulkan drivers | ready (not reduced) |
| BUG-9 | [HeCBench lud wrong results](BUG-9-vk-lud-wrong-results.md) | Vulkan | wrong results on all four Vulkan drivers | ready (not reduced) |
| BUG-10 | [ANV sub-group size](BUG-10-vk-anv-subgroup-size.md) | Vulkan runtime | wrong sub-group results on Intel | fixed on branch `vk/per-kernel-subgroup-size`; file only if the PR needs an issue to reference |
| BUG-11 | [clspv abort on unaligned local-memory GEPs](BUG-11-vk-clspv-unaligned-local-gep.md) | clspv, through the Vulkan backend | scan kernels fail to compile | AdaptiveCpp issue ready; reproduce with an unpatched upstream clspv before filing the clspv issue |
| BUG-12 | [OpenCL accepts `enable_profiling`](BUG-12-ocl-enable-profiling-not-rejected.md) | OpenCL runtime | abort in `get_profiling_info` instead of `errc::feature_not_supported` at queue construction | ready; minor |
| BUG-13 | [OpenCL USM after runtime re-initialisation](BUG-13-ocl-usm-after-runtime-reinit.md) | OpenCL runtime | `malloc_device` returns null, process terminated | ready |
| BUG-14 | [OpenCL `sub_group_sizes` heuristic](BUG-14-ocl-sub-group-sizes.md) | OpenCL runtime | reports 1…512 on the Intel GPU; HeCBench softmax dies with SIGFPE | ready (the heuristic is documented in the code) |
| BUG-15 | [`sycl::native::divide` on Vulkan](BUG-15-vk-native-divide.md) | Vulkan libkernel | the kernel fails to load, program terminates | ready; small enough for a PR |

BUG-5 has no report of its own: it is the HeCBench triage table in
findings.md from which BUG-6 to BUG-11, BUG-13 and BUG-14 were split out.

Suggested filing order, most user-visible first: BUG-8, BUG-9, BUG-11 with
BUG-7, BUG-15, BUG-6, BUG-2, BUG-1, BUG-13, BUG-14, BUG-12, BUG-3, BUG-4;
BUG-10 together with its fix.

## Before filing

1. **Re-run the reproducer on current `develop`.** All results are from
   `f4ff3681` (2026-10-29). If the bug no longer reproduces, do not file it;
   record the fixing commit in findings.md (`Status: fixed (<commit>)`)
   instead. Update *Describe your setup* if you re-ran on another machine.
2. **Search existing issues**, open and closed, for the error message or the
   function name.
3. **Follow the template's AI policy.** The bug report template asks
   reporters not to use AI to speculate on root causes and not to mix AI
   speculation with observed evidence, and maintainers may close reports
   that do. These drafts were written with AI assistance. They are therefore
   limited to the reproducer, its verbatim output, the expected behaviour
   (with specification quotes where relevant), the setup, and code locations
   that were checked by reading the code at `f4ff3681`. The mechanisms and
   proposed fixes in findings.md were left out on purpose; do not add them
   back. Before filing:
   * run each reproducer yourself and compare with the quoted output;
   * check every statement under *Optional additional diagnostic
     information* and *Additional context* against the code, and delete
     anything you cannot confirm yourself;
   * if you have a fix that you have verified, open a PR that links the
     issue instead of describing the fix in the issue.
4. **Use the template.** New issue → *Bug report* (blank issues are
   disabled). Use the suggested title and paste the body from between the two
   horizontal rules. The template's "Important: Acceptable AI usage in bug
   reports" section may be removed, as the template itself says.
5. **Cross-link** related issues: BUG-7 and BUG-11; BUG-11's AdaptiveCpp and
   clspv issues; BUG-10 and its PR.
6. **Record the issue URL** in the report's header table and in the
   finding's **Report** line in findings.md.

## Observed but not filed

Failures seen during the benchmark runs that have no report, and why:

| Benchmark (arguments) | Devices | Symptom | Status |
|---|---|---|---|
| lud (`-s 1024 -v`) | CUDA (M2) | no output after `WG size of kernel = 16 X 16`; killed by the 180 s timeout in both runs | not triaged |
| lud (`-s 1024 -v`) | OpenMP (M1, M2) | 2052 mismatches from (1, 0) on M2 | not triaged; a different pattern from the Vulkan failure (BUG-9) |
| lud (`-s 1024 -v`) | OpenCL CPU (M2) | segfault | not triaged |
| scan (`16777216 100`) | OpenMP (M1, M2) | segfault | not triaged |
| memcpy (`8192`) | OpenMP (M2) | segfault (exit code 139) after printing all results | not triaged |
| layernorm (`8 1024 12288 200`) | RADV (M1) | `Mismatch of out` | may be the sub-group size problem of BUG-10; not checked |
| atomicCost (`16 10`) | RADV, llvmpipe (M1) | 2 of 3 sub-tests fail | the FP64 sub-test is BUG-6; the other failure is not triaged (on M2 only the FP64 sub-test fails) |
| pathfinder (`-DBENCH_PRINT`) | OpenMP (M1) | output differs from HIP, RADV and llvmpipe | did not reproduce on M2 (all 7 devices bit-identical) |
| shuffle (`200000 100`) | ANV and OpenCL on the Intel GPU, llvmpipe, OpenCL CPU, OpenMP | some sub-group-size-specific sub-tests fail | attributed to the benchmark: the same sub-tests fail with OpenCL on the same GPU |
| babelstream (default) | HIP, RADV, OpenMP (M1); ANV, NVIDIA, CUDA (M2) | dot-product check fails | attributed to the benchmark's tolerance: native backends fail too |
| nbody (`16000 30`) | RADV (M1), OpenCL CPU (M2) | `FAIL` although the energy matches the reference to 6 digits | benchmark: absolute tolerance of 1e-3 on a value of ~1.7e5 |
| binomial (default); layernorm; scan | llvmpipe (>600 s); llvmpipe and OpenMP (>180 s); OpenCL CPU (>180 s) | time limit exceeded | slow, not failures |
