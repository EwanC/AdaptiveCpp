// Apply #pragma unroll only when the kernel is JIT-compiled for the Vulkan backend
// (JIT reflection, ACPP_EXT_JIT_COMPILE_IF); prints which loop ran and its median time.
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace jit = sycl::AdaptiveCpp_jit;

inline float sum_unrolled(const float *in, size_t m) {
  float s = 0;
#pragma unroll 8
  for (size_t i = 0; i < m; i++) s += in[i];
  return s;
}

inline float sum_plain(const float *in, size_t m) {
  float s = 0;
  for (size_t i = 0; i < m; i++) s += in[i];
  return s;
}

int main(int argc, char **argv) {
  sycl::queue q{sycl::property::queue::in_order()};
  const size_t m = argc > 99 ? 1 : 65536;
  float *in = sycl::malloc_device<float>(m, q);
  float *sum = sycl::malloc_device<float>(1, q);
  int *which = sycl::malloc_device<int>(1, q);
  std::vector<float> h(m);
  for (size_t i = 0; i < m; i++) h[i] = 1.0f / (1 + i % 7);
  q.memcpy(in, h.data(), m * sizeof(float)).wait();

  auto launch = [&] {
    q.single_task([=] {
      __acpp_if_target_sscp(
        jit::compile_if_else(
          jit::reflect<jit::reflection_query::runtime_backend>() ==
              static_cast<int>(sycl::backend::vk),
          [&]() {
            which[0] = 1;
            sum[0] = sum_unrolled(in, m);
          },
          [&]() {
            which[0] = 0;
            sum[0] = sum_plain(in, m);
          });
      );
    });
  };
  launch();
  q.wait();
  std::vector<double> t;
  for (int r = 0; r < 15; r++) {
    auto t0 = std::chrono::steady_clock::now();
    launch();
    q.wait();
    t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  int w = -1;
  float s = 0;
  q.memcpy(&w, which, sizeof(int));
  q.memcpy(&s, sum, sizeof(float)).wait();
  printf("%s: %s loop, %.3f ms (sum %g)\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
         w == 1 ? "unrolled" : w == 0 ? "plain" : "?", t[t.size() / 2], s);
}
