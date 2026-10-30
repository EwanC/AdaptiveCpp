// nbody acceleration kernel (as in mb.cpp) with three inverse-sqrt spellings,
// to measure how each backend lowers them.
// usage: rsq [n]   (default 16000)
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
using clk = std::chrono::steady_clock;
struct P { float pos[3], vel[3], acc[3], mass; };

template <int V> float inv_sqrt(float x) {
  if constexpr (V == 0) return sycl::rsqrt(x);
  else if constexpr (V == 1) return sycl::native::rsqrt(x);
  else return 1.0f / sycl::sqrt(x);
}

template <int V> double run(sycl::queue &q, P *p, int n) {
  sycl::nd_range<1> ndr{sycl::range<1>((n + 255) / 256 * 256), sycl::range<1>(256)};
  auto k = [&] {
    q.parallel_for(ndr, [=](sycl::nd_item<1> it) {
      int i = it.get_global_id(0);
      if (i >= n) return;
      auto pi = p[i];
      float a0 = 0, a1 = 0, a2 = 0;
      for (int j = 0; j < n; j++) {
        auto pj = p[j];
        float dx = pj.pos[0] - pi.pos[0], dy = pj.pos[1] - pi.pos[1], dz = pj.pos[2] - pi.pos[2];
        float r = inv_sqrt<V>(dx * dx + dy * dy + dz * dz + 1e-3f);
        float s = 6.67259e-11f * pj.mass * r * r * r;
        a0 += dx * s; a1 += dy * s; a2 += dz * s;
      }
      p[i].acc[0] = a0; p[i].acc[1] = a1; p[i].acc[2] = a2;
    });
  };
  k(); q.wait();
  const int R = 5;
  auto t0 = clk::now();
  for (int r = 0; r < R; r++) { k(); q.wait(); }
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count() / R;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 16000;
  sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order()}};
  std::vector<P> hp(n);
  for (int i = 0; i < n; i++) hp[i] = P{{float(i % 97), float(i % 13), float(i % 7)}, {}, {}, 1.f};
  P *p = sycl::malloc_device<P>(n, q);
  q.memcpy(p, hp.data(), n * sizeof(P)).wait();
  double t0 = run<0>(q, p, n), t1 = run<1>(q, p, n), t2 = run<2>(q, p, n);
  printf("%-45s rsqrt %.2f ms  native::rsqrt %.2f ms  1/sqrt %.2f ms\n",
         q.get_device().get_info<sycl::info::device::name>().c_str(), t0, t1, t2);
  sycl::free(p, q);
}
