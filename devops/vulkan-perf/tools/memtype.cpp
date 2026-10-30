// Host-side read/write bandwidth through vkMapMemory for every HOST_VISIBLE
// memory type of every Vulkan device. Explains staging-buffer copy speed
// (the Vulkan backend copies to/from mapped staging memory with memcpy).
// build: g++ -O2 memtype.cpp -I$VULKAN_SDK/include -L$VULKAN_SDK/lib -lvulkan -o memtype
// usage: memtype [MB]   (default 64)
#include <vulkan/vulkan.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using clk = std::chrono::steady_clock;

static std::string flags(VkMemoryPropertyFlags f) {
  std::string s;
  if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) s += "DL|";
  if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) s += "HV|";
  if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) s += "HC|";
  if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) s += "HCACHED|";
  if (!s.empty()) s.pop_back();
  return s;
}

int main(int argc, char **argv) {
  size_t bytes = (argc > 1 ? atol(argv[1]) : 64) << 20;
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst;
  if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return 1;
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> pds(n);
  vkEnumeratePhysicalDevices(inst, &n, pds.data());
  std::vector<char> host(bytes, 1);
  for (auto pd : pds) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice dev;
    if (vkCreateDevice(pd, &dci, nullptr, &dev) != VK_SUCCESS) continue;
    printf("%s\n", props.deviceName);
    for (uint32_t t = 0; t < mp.memoryTypeCount; t++) {
      auto f = mp.memoryTypes[t].propertyFlags;
      if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
      VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      mai.allocationSize = bytes;
      mai.memoryTypeIndex = t;
      VkDeviceMemory mem;
      auto ta = clk::now();
      if (vkAllocateMemory(dev, &mai, nullptr, &mem) != VK_SUCCESS) {
        printf("  type %u %-22s alloc failed\n", t, flags(f).c_str());
        continue;
      }
      void *p;
      vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &p);
      double alloc_map_us = std::chrono::duration<double, std::micro>(clk::now() - ta).count();
      memcpy(p, host.data(), bytes); // first touch
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, mem, 0, VK_WHOLE_SIZE};
      auto t0 = clk::now();
      memcpy(p, host.data(), bytes);
      if (!(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) vkFlushMappedMemoryRanges(dev, 1, &range);
      double w = std::chrono::duration<double>(clk::now() - t0).count();
      t0 = clk::now();
      if (!(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) vkInvalidateMappedMemoryRanges(dev, 1, &range);
      memcpy(host.data(), p, bytes);
      double r = std::chrono::duration<double>(clk::now() - t0).count();
      printf("  type %u heap %u %-22s write %7.2f GB/s  read %7.2f GB/s  (alloc+map %.0f us)\n", t,
             mp.memoryTypes[t].heapIndex, flags(f).c_str(), bytes / w / 1e9, bytes / r / 1e9,
             alloc_map_us);
      vkUnmapMemory(dev, mem);
      vkFreeMemory(dev, mem, nullptr);
      // cost of a small staging-like allocation: allocate, map, unmap, free
      for (size_t small : {size_t(4096), size_t(1) << 20}) {
        const int N = 50;
        mai.allocationSize = small;
        auto t1 = clk::now();
        for (int k = 0; k < N; k++) {
          VkDeviceMemory m2;
          if (vkAllocateMemory(dev, &mai, nullptr, &m2) != VK_SUCCESS) break;
          void *q2;
          vkMapMemory(dev, m2, 0, VK_WHOLE_SIZE, 0, &q2);
          vkUnmapMemory(dev, m2);
          vkFreeMemory(dev, m2, nullptr);
        }
        printf("      alloc+map+unmap+free %7zu B: %8.1f us\n", small,
               std::chrono::duration<double, std::micro>(clk::now() - t1).count() / N);
      }
    }
    vkDestroyDevice(dev, nullptr);
  }
  vkDestroyInstance(inst, nullptr);
}
