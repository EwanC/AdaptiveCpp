# BUG-4: `vk_allocator` pointer lookup is a linear scan with inconsistent bounds

| | |
|---|---|
| File at | <https://github.com/AdaptiveCpp/AdaptiveCpp/issues/new?template=bug_report.md> (label `bug`) — or fix directly in a PR |
| Suggested title | `[VK] vk_allocator::find_user_alloc / query_pointer use a linear scan per lookup` |
| Backend / devices | Vulkan; all drivers |
| Evidence | [findings.md](../vulkan-perf/findings.md) BUG-4 (code inspection only) |
| Note | Minor. No incorrect behaviour has been observed and the cost has not been measured; it is listed because it runs on every `memcpy`/`memset` and the two lookups disagree on the interval bounds. Probably better handled as a small PR than as an issue. |

The issue body follows the repository's bug report template; copy everything
below the line.

---

**Bug summary**

`vk_allocator::find_user_alloc()`, called for every `memcpy` and `memset` on the
Vulkan backend, looks up interior pointers by iterating over all live
allocations, and uses a different interval test than
`vk_allocator::query_pointer()`:

```cpp
// src/runtime/vk/vk_allocator.cpp, find_user_alloc()
for (auto &alloc : _allocs) {
  ...
  if (ptr > base && ptr < end) {        // exclusive lower bound
    return &alloc_info;
  }
}

// src/runtime/vk/vk_allocator.cpp, query_pointer()
for (const auto &alloc : _allocs) {
  ...
  if (dev_ptr >= base && dev_ptr < end) {  // inclusive lower bound
```

The exclusive lower bound does not cause wrong results today, because
`find_user_alloc()` checks `_allocs.count(ptr)` for exact base pointers first.

**To Reproduce**

Code inspection at `f4ff3681`; `src/runtime/vk/vk_allocator.cpp`,
`find_user_alloc()` and `query_pointer()`. Every
`q.memcpy(dst, src, n)` / `q.memset(ptr, v, n)` reaches `find_user_alloc()`
through `vk_queue::find_or_create_allocation()` / `vk_queue::submit_memset()`;
an operation on an interior pointer (`ptr + offset`) of an allocation takes
the linear path.

**Expected behavior**

Constant- or logarithmic-time lookup (e.g. an ordered map keyed by base address
with `upper_bound`), and the same interval semantics (`base <= ptr < base + size`)
in both functions.

**Describe your setup**

* AdaptiveCpp `develop` at `f4ff3681` ("[VK] Create pass to legalize int
  widths for clspv (#2272)"), Vulkan backend
* Not hardware specific

**Optional additional diagnostic information**

Not applicable.

**Additional context**

`_allocs` is a `std::unordered_map<vk::DeviceAddress, vk_alloc_info>`
(`include/hipSYCL/runtime/vk/vk_allocator.hpp`), so ordered lookups need a
different container or an additional index.

---
