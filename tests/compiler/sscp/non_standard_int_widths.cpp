// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

// Tests data types whose LLVM IR representation can end up using integer
// types of non-standard bit width after optimization, e.g. `marray<short, 3>`
// is merged into a single `i48` value by SROA/instcombine. Such types are not
// supported by all backends, e.g. clspv cannot lower them to SPIR-V.

#include "common.hpp"
#include <iostream>

int main() {
  sycl::queue q = get_queue();

  constexpr size_t size = 4;

  using short3 = sycl::marray<short, 3>;
  using char3 = sycl::marray<char, 3>;

  short3 *short_ptr = sycl::malloc_shared<short3>(size, q);
  char3 *char_ptr = sycl::malloc_shared<char3>(size, q);
  sycl::vec<float, 3> *vec_ptr =
      sycl::malloc_shared<sycl::vec<float, 3>>(size, q);

  for (size_t i = 0; i < size; ++i) {
    short_ptr[i] = short3{static_cast<short>(i), static_cast<short>(i + 1),
                          static_cast<short>(i + 2)};
    char_ptr[i] = char3{static_cast<char>(i), static_cast<char>(i + 1),
                        static_cast<char>(i + 2)};
    vec_ptr[i] = sycl::vec<float, 3>{static_cast<float>(i),
                                     static_cast<float>(i + 1),
                                     static_cast<float>(i + 2)};
  }

  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for(sycl::range<1>(size), [=](sycl::id<1> id) {
       short3 s = short_ptr[id];
       short3 s_result;
       for (size_t i = 0; i < 3; ++i)
         s_result[i] = static_cast<short>(s[i] * 2);
       short_ptr[id] = s_result;

       char3 c = char_ptr[id];
       char3 c_result;
       for (size_t i = 0; i < 3; ++i)
         c_result[i] = static_cast<char>(c[i] + 1);
       char_ptr[id] = c_result;

       vec_ptr[id] = vec_ptr[id] * 2.f;
     });
   }).wait();

  // CHECK: 0 2 4 | 1 2 3 | 0 2 4
  // CHECK: 2 4 6 | 2 3 4 | 2 4 6
  // CHECK: 4 6 8 | 3 4 5 | 4 6 8
  // CHECK: 6 8 10 | 4 5 6 | 6 8 10
  for (size_t i = 0; i < size; ++i) {
    std::cout << short_ptr[i][0] << " " << short_ptr[i][1] << " "
              << short_ptr[i][2] << " | " << static_cast<int>(char_ptr[i][0])
              << " " << static_cast<int>(char_ptr[i][1]) << " "
              << static_cast<int>(char_ptr[i][2]) << " | " << vec_ptr[i][0]
              << " " << vec_ptr[i][1] << " " << vec_ptr[i][2] << std::endl;
  }

  sycl::free(short_ptr, q);
  sycl::free(char_ptr, q);
  sycl::free(vec_ptr, q);

  return 0;
}
