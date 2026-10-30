// Compare SYCL event profiling against wall clock for one kernel (BUG-1).
// Every event is waited on before the next submission (avoids BUG-2).
// usage: prof [MB]   (default 64)
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
using clk = std::chrono::steady_clock;

int main(int argc, char **argv) {
  size_t n = (argc > 1 ? atol(argv[1]) : 64) * 1024 * 1024 / 4;
  sycl::queue q{sycl::default_selector_v,
                {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};
  float *a = sycl::malloc_device<float>(n, q), *b = sycl::malloc_device<float>(n, q);
  q.fill(a, 1.f, n).wait();
  auto k = [&] {
    return q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
      float x = a[i];
      for (int r = 0; r < 64; r++) x = x * 1.0001f + 0.5f;
      b[i] = x;
    });
  };
  k().wait();
  const int R = 10;
  double wall = 0, prof = 0;
  for (int r = 0; r < R; r++) {
    auto t0 = clk::now();
    auto e = k();
    e.wait();
    wall += std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    prof += (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
             e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-3;
  }
  printf("%-45s wall %.1f us  profiled %.1f us  wall/profiled %.2f\n",
         q.get_device().get_info<sycl::info::device::name>().c_str(), wall / R, prof / R,
         wall / prof);
  sycl::free(a, q);
  sycl::free(b, q);
}
