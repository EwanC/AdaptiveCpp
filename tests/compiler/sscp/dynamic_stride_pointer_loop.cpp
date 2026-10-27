// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: env ACPP_ADAPTIVITY_LEVEL=0 %t | FileCheck %s

#include "common.hpp"
#include <iostream>
#include <vector>

// Each work-item sums the elements of its work-group's segment with a stride
// of the work-group size, mirroring the loop of the generic joint_reduce()
// implementation.
//
// At ACPP_ADAPTIVITY_LEVEL=0 the work-group size is not known when the kernel
// is compiled, so the stride stays a runtime value and LLVM forms a pointer
// induction variable advanced by `getelementptr i8, ptr %p, i64 %bytes`.
// clspv's SimplifyPointerBitcastPass used to rewrite this GEP back and forth
// between i8 and i32 forever, hanging the JIT compilation.

template <class Ptr>
int strided_sum(sycl::group<1> g, Ptr first, Ptr last) {
  const size_t lrange = g.get_local_range().size();
  const size_t num_elements = last - first;
  if (num_elements == 0)
    return 0;
  if (num_elements == 1)
    return *first;

  Ptr start = first + g.get_local_linear_id();
  int sum = 0;
  if (start < last)
    sum = *start;
  for (Ptr p = start + lrange; p < last; p += lrange)
    sum += *p;
  return sum;
}

int main() {
  sycl::queue q = get_queue();

  constexpr size_t local_size = 16;
  constexpr size_t global_size = 64;
  std::vector<int> data(3 * global_size);
  for (size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<int>(i);

  {
    sycl::buffer<int> buf{data.data(), sycl::range<1>{data.size()}};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc{buf, cgh, sycl::read_write};
      cgh.parallel_for(sycl::nd_range<1>{global_size, local_size},
                       [=](sycl::nd_item<1> item) {
        auto g = item.get_group();
        size_t gid = item.get_global_linear_id();
        size_t lsize = g.get_local_range().size();
        auto first = acc.get_pointer() + (gid / lsize) * lsize * 2;
        auto last = first + lsize * 2;
        acc[gid + 2 * global_size] = strided_sum(g, first.get(), last.get());
      });
    });
  }

  long long total = 0;
  for (size_t i = 2 * global_size; i < 3 * global_size; ++i)
    total += data[i];
  // Sum of 0 .. 2 * global_size - 1
  // CHECK: 8128
  std::cout << total << std::endl;
}
