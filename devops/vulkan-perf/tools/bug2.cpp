// BUG-2 reproducer: profiling + in-order queue, events dropped before the
// command completes. Run with VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation.
#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
  sycl::queue q{sycl::default_selector_v,
                {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};
  printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  int *d = sycl::malloc_device<int>(1 << 20, q);
  for (int i = 0; i < 2000; i++)
    q.parallel_for(sycl::range<1>{1 << 20}, [=](sycl::id<1> i) { d[i] = d[i] * 3 + 1; });
  q.wait();
  sycl::free(d, q);
  printf("done\n");
}
