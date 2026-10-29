#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
using clk=std::chrono::steady_clock;
int main(int argc,char**argv){
  size_t n = (argc>1? atol(argv[1]) : 64) * 1024*1024/4;
  sycl::queue q{sycl::default_selector_v,{sycl::property::queue::in_order()}};
  float*a=sycl::malloc_device<float>(n,q),*b=sycl::malloc_device<float>(n,q);
  q.fill(a,1.f,n); q.fill(b,0.f,n); q.wait();
  auto k=[&]{ q.parallel_for(sycl::nd_range<1>{n,256},[=](sycl::nd_item<1> it){ size_t i=it.get_global_id(0); b[i]=a[i]*2.f+1.f;});};
  for(int i=0;i<3;i++)k(); q.wait();
  int R=50; auto t0=clk::now(); for(int i=0;i<R;i++)k(); q.wait();
  double s=std::chrono::duration<double>(clk::now()-t0).count()/R;
  printf("%-45s n=%zuMB  %.1f us/kernel  %.1f GB/s\n", q.get_device().get_info<sycl::info::device::name>().c_str(), n*4>>20, s*1e6, 2.0*n*4/s/1e9);
}
