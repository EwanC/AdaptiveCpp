#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}
struct P { float pos[3], vel[3], acc[3], mass; };

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 16000;
  sycl::queue q{sycl::default_selector_v,
                {sycl::property::queue::in_order()}};
  printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  int *d = sycl::malloc_device<int>(1024, q);
  int h[1024];
  // warmup
  for (int i = 0; i < 20; i++) q.parallel_for(sycl::range<1>{256}, [=](sycl::id<1> i) { d[i] = i; });
  q.memcpy(h, d, 4).wait();
  const int N = 500;

  auto t0 = clk::now();
  for (int i = 0; i < N; i++) q.parallel_for(sycl::range<1>{256}, [=](sycl::id<1> i) { d[i] = i; });
  auto t1 = clk::now();
  q.wait();
  auto t2 = clk::now();
  printf("empty kernel x%d: submit %.2f us/launch, end-to-end %.2f us/launch\n", N, us(t0, t1) / N, us(t0, t2) / N);

  t0 = clk::now();
  for (int i = 0; i < N; i++) q.parallel_for(sycl::range<1>{256}, [=](sycl::id<1> i) { d[i] = i; }).wait();
  t1 = clk::now();
  printf("empty kernel+wait: %.2f us/iter\n", us(t0, t1) / N);

  t0 = clk::now();
  for (int i = 0; i < N; i++) q.memcpy(h, d, 4).wait();
  t1 = clk::now();
  printf("D2H 4B memcpy+wait: %.2f us/iter\n", us(t0, t1) / N);
  t0 = clk::now();
  for (int i = 0; i < N; i++) q.memcpy(d, h, 4).wait();
  t1 = clk::now();
  printf("H2D 4B memcpy+wait: %.2f us/iter\n", us(t0, t1) / N);

  std::vector<P> hp(n);
  for (int i = 0; i < n; i++) hp[i] = P{{float(i % 97), float(i % 13), float(i % 7)}, {}, {}, 1.f};
  P *p = sycl::malloc_device<P>(n, q);
  float *e = sycl::malloc_device<float>(n, q);
  t0 = clk::now();
  q.memcpy(p, hp.data(), n * sizeof(P)).wait();
  t1 = clk::now();
  printf("H2D %zu KB: %.2f us\n", n * sizeof(P) / 1024, us(t0, t1));

  sycl::nd_range<1> ndr{sycl::range<1>((n + 255) / 256 * 256), sycl::range<1>(256)};
  auto acc_k = [&] {
    return q.parallel_for(ndr, [=](sycl::nd_item<1> it) {
      int i = it.get_global_id(0);
      if (i >= n) return;
      auto pi = p[i];
      float a0 = pi.acc[0], a1 = pi.acc[1], a2 = pi.acc[2];
      for (int j = 0; j < n; j++) {
        auto pj = p[j];
        float dx = pj.pos[0] - pi.pos[0], dy = pj.pos[1] - pi.pos[1], dz = pj.pos[2] - pi.pos[2];
        float r = sycl::rsqrt(dx * dx + dy * dy + dz * dz + 1e-3f);
        float s = 6.67259e-11f * pj.mass * r * r * r;
        a0 += dx * s; a1 += dy * s; a2 += dz * s;
      }
      pi.acc[0] = a0; pi.acc[1] = a1; pi.acc[2] = a2;
      p[i] = pi;
    });
  };
  auto upd_k = [&] {
    return q.parallel_for(ndr, [=](sycl::nd_item<1> it) {
      auto i = it.get_global_id(0);
      if (i >= n) return;
      auto pi = p[i];
      for (int k = 0; k < 3; k++) { pi.vel[k] += pi.acc[k] * 0.1f; pi.pos[k] += pi.vel[k] * 0.1f; pi.acc[k] = 0; }
      e[i] = pi.mass * (pi.vel[0] * pi.vel[0] + pi.vel[1] * pi.vel[1] + pi.vel[2] * pi.vel[2]);
      p[i] = pi;
    });
  };
  auto red_k = [&] {
    return q.single_task([=] { for (int i = 1; i < n; i++) e[0] += e[i]; });
  };
  auto timeit = [&](auto f, int reps) {
    f(); q.wait();
    auto a = clk::now();
    for (int r = 0; r < reps; r++) { f(); q.wait(); }
    return us(a, clk::now()) / reps;
  };
  printf("accel kernel+wait: %.1f us\n", timeit(acc_k, 5));
  printf("update kernel+wait: %.1f us\n", timeit(upd_k, 20));
  printf("single_task reduce+wait: %.1f us\n", timeit(red_k, 20));
  float hv;
  printf("step(3 kernels)+wait+D2H 4B: %.1f us\n", timeit([&]{acc_k();upd_k();red_k();q.wait();q.memcpy(&hv,e,4);}, 10));
  return 0;
}
