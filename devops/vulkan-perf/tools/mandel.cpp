// HeCBench mandelbrot kernel in isolation: (a) one allocation reused,
// (b) a fresh malloc_device per launch as HeCBench does. The iteration
// count is a runtime value, as in HeCBench (a constant changes codegen).
// usage: mandel [max_iterations]   (default 100)
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
using clk = std::chrono::steady_clock;
constexpr int rows = 1080, cols = 1920;
int max_it = 100;

void launch(sycl::queue &q, int *out) {
  int mi = max_it;
  q.parallel_for(sycl::nd_range<2>({(rows + 15) / 16 * 16, (cols + 15) / 16 * 16}, {16, 16}),
                 [=](sycl::nd_item<2> it) {
    int i = it.get_global_id(0), j = it.get_global_id(1);
    if (i >= rows || j >= cols) return;
    float cr = -1.5f + i * (2.0f / rows), ci = -1.0f + j * (2.0f / cols);
    float zr = 0, zi = 0;
    int count = 0;
    for (int k = 0; k < mi; ++k) {
      float r = zr, im = zi;
      if (r * r + im * im >= 4.0f) break;
      zr = r * r - im * im + cr;
      zi = 2 * r * im + ci;
      count++;
    }
    out[i * cols + j] = count;
  }).wait();
}

int main(int argc, char **argv) {
  if (argc > 1) max_it = atoi(argv[1]);
  sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order()}};
  int *d = sycl::malloc_device<int>(rows * cols, q);
  launch(q, d);
  const int R = 20;
  auto t0 = clk::now();
  for (int r = 0; r < R; r++) launch(q, d);
  double reuse = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / R;
  sycl::free(d, q);
  double fresh = 0;
  for (int r = 0; r < R; r++) {
    int *f = sycl::malloc_device<int>(rows * cols, q);
    auto t1 = clk::now();
    launch(q, f);
    fresh += std::chrono::duration<double, std::milli>(clk::now() - t1).count();
    sycl::free(f, q);
  }
  printf("%-45s kernel+wait: reused alloc %.3f ms  fresh alloc %.3f ms\n",
         q.get_device().get_info<sycl::info::device::name>().c_str(), reuse, fresh / R);
}
