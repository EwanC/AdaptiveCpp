// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: env ACPP_ADAPTIVITY_LEVEL=0 %t | FileCheck %s

// At ACPP_ADAPTIVITY_LEVEL=0 the double kernel is part of the compiled module
// even if it is never launched, which requires fp64 support from the device.
// MoltenVK does not support fp64.
// UNSUPPORTED: system-darwin

#include "common.hpp"
#include <cmath>
#include <iostream>
#include <limits>

// clspv lowers the OpenCL signbit(double) builtin to an invalid i64 -> i32
// zext and crashes. The Vulkan backend reports no fp64 support, so the double
// kernel below is never launched there. However, at ACPP_ADAPTIVITY_LEVEL=0
// all kernels of the translation unit are compiled into one module, and the
// crash used to break the float kernel as well.

template <class T> bool test_signbit(sycl::queue &q) {
  constexpr int n = 8;
  const T inf = std::numeric_limits<T>::infinity();
  const T in[n] = {T(1), T(-1), T(0), -T(0), -inf, inf, T(-2.5), T(3)};

  T *d_in = sycl::malloc_device<T>(n, q);
  int *d_out = sycl::malloc_device<int>(n, q);
  q.copy(in, d_in, n).wait();
  q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
     d_out[i] = sycl::signbit(d_in[i]) ? 1 : 0;
   }).wait();

  int out[n];
  q.copy(d_out, out, n).wait();
  sycl::free(d_in, q);
  sycl::free(d_out, q);

  bool ok = true;
  for (int i = 0; i < n; ++i)
    ok &= (out[i] != 0) == std::signbit(in[i]);
  return ok;
}

int main() {
  sycl::queue q = get_queue();

  // CHECK: float: PASS
  std::cout << "float: " << (test_signbit<float>(q) ? "PASS" : "FAIL")
            << std::endl;

  if (q.get_device().has(sycl::aspect::fp64))
    std::cout << "double: " << (test_signbit<double>(q) ? "PASS" : "FAIL")
              << std::endl;
  // CHECK-NOT: FAIL
}
