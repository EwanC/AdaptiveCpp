// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: env ACPP_ADAPTIVITY_LEVEL=0 %t | FileCheck %s

#include "common.hpp"
#include <cstdint>
#include <iostream>

// clspv fails on 64-bit atomic_fetch_min. The 64-bit kernel is only launched
// on devices with atomic64 support, which excludes the Vulkan backend, but at
// ACPP_ADAPTIVITY_LEVEL=0 it is compiled into the same module as the 32-bit
// kernel. Kernels clspv fails on must not break the other kernels of the
// module.

template <class T> T fetch_min(sycl::queue &q) {
  T *ptr = sycl::malloc_device<T>(1, q);
  T init = 100;
  q.copy(&init, ptr, 1).wait();
  q.parallel_for(sycl::range<1>{64}, [=](sycl::id<1> i) {
     sycl::atomic_ref<T, sycl::memory_order::relaxed,
                      sycl::memory_scope::device>
         r{*ptr};
     r.fetch_min(static_cast<T>(i[0]) + 1);
   }).wait();

  T result = 0;
  q.copy(ptr, &result, 1).wait();
  sycl::free(ptr, q);
  return result;
}

int main() {
  sycl::queue q = get_queue();

  // CHECK: 32-bit: 1
  std::cout << "32-bit: " << fetch_min<int32_t>(q) << std::endl;

  if (q.get_device().has(sycl::aspect::atomic64))
    std::cout << "64-bit: " << fetch_min<int64_t>(q) << std::endl;
  else
    std::cout << "64-bit: skipped" << std::endl;
  // CHECK: 64-bit: {{1|skipped}}
}
