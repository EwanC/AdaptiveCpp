// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: env ACPP_ADAPTIVITY_LEVEL=0 %t | FileCheck %s

#include "common.hpp"
#include <iostream>
#include <vector>

// Work-item 0 of each work-group passes a value to the other work-items
// through local memory holding a single int. The kernel also queries the
// maximum sub-group size, for which clspv creates a module scope int variable
// in another address space. When the local memory is a module scope int
// variable as well, which is the case at the default adaptivity level, clspv's
// ShareModuleScopeVariablesPass used to try to merge the two variables despite
// their different address spaces and crashed.

int main() {
  sycl::queue q = get_queue();

  constexpr size_t local_size = 16;
  constexpr size_t global_size = 64;
  int *out = sycl::malloc_device<int>(global_size, q);
  unsigned *max_sg_size = sycl::malloc_device<unsigned>(1, q);

  q.submit([&](sycl::handler &cgh) {
     sycl::local_accessor<int, 1> shared{1, cgh};
     cgh.parallel_for(sycl::nd_range<1>{global_size, local_size},
                      [=](sycl::nd_item<1> item) {
                        if (item.get_local_id(0) == 0)
                          shared[0] = static_cast<int>(item.get_global_id(0));
                        sycl::group_barrier(item.get_group());
                        out[item.get_global_id(0)] = shared[0];

                        if (item.get_global_id(0) == 0)
                          *max_sg_size =
                              item.get_sub_group().get_max_local_range()[0];
                      });
   }).wait();

  std::vector<int> result(global_size);
  q.copy(out, result.data(), global_size).wait();
  unsigned sg_size = 0;
  q.copy(max_sg_size, &sg_size, 1).wait();

  bool ok = sg_size > 0;
  for (size_t i = 0; i < global_size; ++i)
    ok &= result[i] == static_cast<int>((i / local_size) * local_size);
  // CHECK: PASS
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  sycl::free(out, q);
  sycl::free(max_sg_size, q);
}
