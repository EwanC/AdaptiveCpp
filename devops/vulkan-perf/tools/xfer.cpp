// Host<->device memcpy sweep with pageable host memory (IMPR-3).
// For each size: one warm-up copy, then reps copies each followed by wait().
// usage: xfer [max_MB]   (default 256)
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
using clk = std::chrono::steady_clock;

int main(int argc, char **argv) {
  size_t max_bytes = (argc > 1 ? atol(argv[1]) : 256) << 20;
  sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order()}};
  printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  char *d = sycl::malloc_device<char>(max_bytes, q);
  std::vector<char> h(max_bytes, 1);
  printf("%10s %12s %10s %12s %10s\n", "bytes", "H2D us", "H2D GB/s", "D2H us", "D2H GB/s");
  for (size_t sz = 4; sz <= max_bytes; sz *= 4) {
    int reps = sz <= (1 << 16) ? 200 : sz <= (16 << 20) ? 20 : 5;
    q.memcpy(d, h.data(), sz).wait();
    auto t0 = clk::now();
    for (int r = 0; r < reps; r++) q.memcpy(d, h.data(), sz).wait();
    double h2d = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / reps;
    q.memcpy(h.data(), d, sz).wait();
    t0 = clk::now();
    for (int r = 0; r < reps; r++) q.memcpy(h.data(), d, sz).wait();
    double d2h = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / reps;
    printf("%10zu %12.1f %10.2f %12.1f %10.2f\n", sz, h2d, sz / h2d / 1e3, d2h, sz / d2h / 1e3);
  }
  sycl::free(d, q);
}
