// User-guidance checks for the Vulkan backend (wall clock, median of reps):
//   chain kernels: 1024-step dependent chains per work-item
//     mul+add vs sycl::fma        (IMPR-5: no FMA contraction)
//     a / x   vs a * native::recip (IMPR-11: precise division sequence)
//   serial loop: single_task sum, plain vs #pragma unroll 8 vs manual x8 (IMPR-12)
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

template <class F> double median_ms(sycl::queue &q, F f, int reps = 15) {
  f();
  q.wait();
  std::vector<double> t;
  for (int r = 0; r < reps; r++) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    q.wait();
    t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(int argc, char **argv) {
  sycl::queue q{sycl::property::queue::in_order()};
  const size_t n = argc > 99 ? 1 : 4 << 20;
  const int steps = argc > 99 ? 1 : 1024;
  const size_t m = argc > 99 ? 1 : 65536;
  float a = argc > 99 ? 2.0f : 1.0001f, b = argc > 99 ? 1.0f : 0.5f;
  float da = argc > 99 ? 2.0f : 1.5f, db = argc > 99 ? 1.0f : 0.75f;
  float *out = sycl::malloc_device<float>(n, q);
  float *in = sycl::malloc_device<float>(m, q);
  float *sum = sycl::malloc_device<float>(1, q);
  std::vector<float> h(m);
  for (size_t i = 0; i < m; i++) h[i] = 1.0f / (1 + i % 7);
  q.memcpy(in, h.data(), m * sizeof(float)).wait();

  auto check = [&] {
    float r[3];
    q.memcpy(&r[0], out + 1, sizeof(float));
    q.memcpy(&r[1], out + n / 2, sizeof(float));
    q.memcpy(&r[2], out + n - 1, sizeof(float)).wait();
    return r[0] + r[1] + r[2];
  };
  auto checksum = [&] {
    float r = 0;
    q.memcpy(&r, sum, sizeof(float)).wait();
    return r;
  };

  printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  double t = median_ms(q, [&] {
    q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
      float x = i[0] * 1e-7f;
      for (int s = 0; s < steps; s++) x = x * a + b;
      out[i] = x;
    });
  });
  printf("  chain x*a+b               %9.3f ms  (check %g)\n", t, check());

  t = median_ms(q, [&] {
    q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
      float x = i[0] * 1e-7f;
      for (int s = 0; s < steps; s++) x = sycl::fma(x, a, b);
      out[i] = x;
    });
  });
  printf("  chain sycl::fma(x,a,b)    %9.3f ms  (check %g)\n", t, check());

  t = median_ms(q, [&] {
    q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
      float x = i[0] * 1e-7f;
      for (int s = 0; s < steps / 8; s++) x = da / (x + db);
      out[i] = x;
    });
  });
  printf("  chain a/(x+b)             %9.3f ms  (check %g)\n", t, check());

  t = median_ms(q, [&] {
    q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
      float x = i[0] * 1e-7f;
      for (int s = 0; s < steps / 8; s++) x = da * sycl::native::recip(x + db);
      out[i] = x;
    });
  });
  printf("  chain a*native::recip(x+b) %8.3f ms  (check %g)\n", t, check());

  t = median_ms(q, [&] {
    q.single_task([=] {
      float s = 0;
      for (size_t i = 0; i < m; i++) s += in[i];
      sum[0] = s;
    });
  });
  printf("  serial sum, plain         %9.3f ms  (sum %g)\n", t, checksum());

  t = median_ms(q, [&] {
    q.single_task([=] {
      float s = 0;
#pragma unroll 8
      for (size_t i = 0; i < m; i++) s += in[i];
      sum[0] = s;
    });
  });
  printf("  serial sum, pragma unroll %9.3f ms  (sum %g)\n", t, checksum());

  t = median_ms(q, [&] {
    q.single_task([=] {
      float s = 0;
      size_t i = 0;
      for (; i + 8 <= m; i += 8) {
        float v0 = in[i], v1 = in[i + 1], v2 = in[i + 2], v3 = in[i + 3];
        float v4 = in[i + 4], v5 = in[i + 5], v6 = in[i + 6], v7 = in[i + 7];
        s += v0; s += v1; s += v2; s += v3; s += v4; s += v5; s += v6; s += v7;
      }
      for (; i < m; i++) s += in[i];
      sum[0] = s;
    });
  });
  printf("  serial sum, manual x8     %9.3f ms  (sum %g)\n", t, checksum());

  sycl::free(out, q);
  sycl::free(in, q);
  sycl::free(sum, q);
}
