#include <sycl/sycl.hpp>
int main(int c,char**v){ int K=atoi(v[1]),M=atoi(v[2]); sycl::queue q{sycl::default_selector_v,{sycl::property::queue::in_order()}};
 int*d=sycl::malloc_device<int>(256,q); int h[4];
 for(int i=0;i<K;i++){ q.parallel_for(sycl::range<1>{256},[=](sycl::id<1> i){d[i]=i;}); if(i%100==0) q.wait(); }
 q.wait();
 for(int i=0;i<M;i++) q.memcpy(h,d,4).wait();
}
