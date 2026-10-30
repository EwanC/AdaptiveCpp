// Index-decomposition throughput (IMPR-6): the su3 pattern
// i = t/36, j = (t%36)/9, k = (t%9)/3, l = t%3, evaluated with 64-bit
// (size_t) and 32-bit (unsigned) arithmetic.
// usage: idx [n_millions] [iters]   (default 16, 64)
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
using clk = std::chrono::steady_clock;

template <class T> double run(sycl::queue &q, unsigned *out, size_t n, int iters) {
  auto k = [&] {
    q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> id) {
      T v = id[0];
      unsigned acc = 0;
      for (int r = 0; r < iters; r++) {
        T t = v + T(r);
        acc += unsigned(t / 36 + (t % 36) / 9 + (t % 9) / 3 + t % 3);
      }
      out[id[0]] = acc;
    });
  };
  k(); q.wait();
  const int R = 10;
  auto t0 = clk::now();
  for (int r = 0; r < R; r++) k();
  q.wait();
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count() / R;
}

int main(int argc, char **argv) {
  size_t n = (argc > 1 ? atol(argv[1]) : 16) << 20;
  int iters = argc > 2 ? atoi(argv[2]) : 64;
  sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order()}};
  unsigned *out = sycl::malloc_device<unsigned>(n, q);
  unsigned h[2];
  double t64 = run<size_t>(q, out, n, iters);
  q.memcpy(h, out + n / 2, 4).wait();
  double t32 = run<unsigned>(q, out, n, iters);
  q.memcpy(h + 1, out + n / 2, 4).wait();
  printf("%-45s n=%zuM iters=%d  size_t %.2f ms  unsigned %.2f ms  ratio %.2f  (check %u %u)\n",
         q.get_device().get_info<sycl::info::device::name>().c_str(), n >> 20, iters,
         t64, t32, t64 / t32, h[0], h[1]);
  sycl::free(out, q);
}
