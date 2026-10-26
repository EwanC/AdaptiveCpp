/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include "../hardware.hpp"
#include "hipSYCL/runtime/vk/vk_native_objects.hpp"
#include <mutex>
#include <unordered_map>
#include <vulkan/vulkan_raii.hpp>

namespace hipsycl {
namespace rt {

class vk_allocator;

namespace vk_device_features {
// Bit for each SPIR-V capability feature we may encounter in kernel code
// and can lazily check against device support.
enum feature_bits {
  shaderInt8 = 1 << 0,
  shaderInt16 = 1 << 1,
  shaderInt64 = 1 << 2,
  shaderFloat16 = 1 << 3,
  storagePushConstant8 = 1 << 4,
  storagePushConstant16 = 1 << 5,
  variablePointers = 1 << 6,
  variablePointersStorageBuffer = 1 << 7,
  groupNonUniform = 1 << 8,
  groupNonUniformShuffle = 1 << 9,
  groupNonUniformVote = 1 << 10,
  shaderFloat64 = 1 << 11,
  uniformAndStorageBuffer16BitAccess = 1 << 12,
  uniformAndStorageBuffer8BitAccess = 1 << 13
};
}; // namespace vk_device_features

// Bit for each extension we can enable when creating a logical device,
// improves extension querying compared to C string comparisons against
// extension names.
namespace vk_device_extensions {
enum extension_bits : uint16_t {
  khr_portability_subset = 1 << 0,
  khr_calibrated_timestamps = 1 << 1,
  ext_calibrated_timestamps = 1 << 2,
  khr_external_memory_fd = 1 << 3,
  khr_external_semaphore_fd = 1 << 4,
  khr_external_memory_win32 = 1 << 5,
  khr_external_semaphore_win32 = 1 << 6,
};
} // namespace vk_device_extensions

// The Vulkan handle type bits matching vk_native_external_handle_type. Objects
// the runtime creates as exportable use these, as no implementation supports
// both opaque file descriptors and Win32 handles.
#ifdef VK_USE_PLATFORM_WIN32_KHR
inline constexpr vk::ExternalMemoryHandleTypeFlagBits
    vk_native_external_memory_handle_bit =
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
inline constexpr vk::ExternalSemaphoreHandleTypeFlagBits
    vk_native_external_semaphore_handle_bit =
        vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
#else
inline constexpr vk::ExternalMemoryHandleTypeFlagBits
    vk_native_external_memory_handle_bit =
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
inline constexpr vk::ExternalSemaphoreHandleTypeFlagBits
    vk_native_external_semaphore_handle_bit =
        vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
#endif

class vk_hardware_context : public hardware_context,
                            public vk_native_device_provider,
                            public vk_native_external_semaphore_provider {
public:
  vk_hardware_context(const vk::raii::PhysicalDevice &, VkInstance instance,
                      int dev_id, uint16_t features);
  vk_hardware_context(vk_hardware_context const &) = delete;
  vk_hardware_context(vk_hardware_context &&) = default;

  bool is_cpu() const override;
  bool is_gpu() const override;

  std::size_t get_max_kernel_concurrency() const override;
  std::size_t get_max_memcpy_concurrency() const override;

  std::string get_device_name() const override;
  std::string get_vendor_name() const override;
  std::string get_device_arch() const override;

  bool has(device_support_aspect aspect) const override;
  std::size_t get_property(device_uint_property prop) const override;

  std::vector<std::size_t>
  get_property(device_uint_list_property prop) const override;

  std::string get_driver_version() const override;
  std::string get_profile() const override;

  std::size_t get_platform_index() const override;

  vk_allocator *get_allocator();
  void init();

  vk_device_native_handle get_native_device_handle() const override;

  bool
  has_external_semaphore_support(vk_external_handle_type type) const override;
  bool export_semaphore_handle(VkSemaphore semaphore,
                               vk_external_handle_type type,
                               vk_external_handle &out) const override;
  VkSemaphore
  import_timeline_semaphore(const vk_external_handle &handle) override;
  void destroy_imported_semaphore(VkSemaphore semaphore) override;

  bool has_external_memory_support(vk_external_handle_type type) const {
    const uint16_t bit = external_memory_extension_bit(type);
    return bit != 0 && are_extensions_enabled(bit);
  }

  // Extension bit providing the given external memory handle type, or 0 if
  // the handle type is not supported by this platform at all.
  static uint16_t external_memory_extension_bit(vk_external_handle_type type) {
    if (type == vk_external_handle_type::opaque_fd)
      return vk_device_extensions::khr_external_memory_fd;
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (type == vk_external_handle_type::opaque_win32)
      return vk_device_extensions::khr_external_memory_win32;
#endif
    return 0;
  }

  // Extension bit providing the given external semaphore handle type, or 0 if
  // the handle type is not supported by this platform at all.
  static uint16_t
  external_semaphore_extension_bit(vk_external_handle_type type) {
    if (type == vk_external_handle_type::opaque_fd)
      return vk_device_extensions::khr_external_semaphore_fd;
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (type == vk_external_handle_type::opaque_win32)
      return vk_device_extensions::khr_external_semaphore_win32;
#endif
    return 0;
  }

  const vk::raii::Device &get_device() const { return _device; }
  const vk::raii::PhysicalDevice &get_physical_device() const {
    return _physical_device;
  }
  vk::raii::Queue &get_queue() { return _queue; }
  uint32_t get_queue_index() const { return _queue_index; }
  uint32_t get_max_push_constant_size() const {
    return _limits.maxPushConstantsSize;
  }
  uint32_t get_max_uniform_buffer_range() const {
    return _limits.maxUniformBufferRange;
  }
  uint32_t get_min_uniform_buffer_offset_alignment() const {
    return _limits.minUniformBufferOffsetAlignment;
  }
  uint32_t get_subgroup_size() const { return _subgroup_size; }
  uint16_t get_phys_dev_features() const { return _physical_dev_features; }

  bool are_extensions_enabled(uint16_t bits) const {
    return (bits & _enabled_extensions) == bits;
  }

private:
  size_t global_mem_size() const;

  vk::raii::PhysicalDevice _physical_device = nullptr;
  VkInstance _instance = VK_NULL_HANDLE; // Non-owning
  vk::raii::Device _device = nullptr;
  uint32_t _queue_index = UINT32_MAX;
  vk::raii::Queue _queue = nullptr;
  vk::PhysicalDeviceProperties _properties{};
  vk::PhysicalDeviceLimits _limits{};
  size_t _subgroup_size{};
  size_t _max_num_subgroups{};
  size_t _max_alloc_size{};

  int _dev_id;
  // Semaphores created on this device from imported external payloads, owned
  // by the runtime on behalf of the user until explicitly destroyed. Held
  // indirectly so that vk_hardware_context remains movable.
  struct imported_semaphore_store {
    std::unordered_map<VkSemaphore, vk::raii::Semaphore> _semaphores;
    std::mutex _mutex;
  };
  std::unique_ptr<imported_semaphore_store> _imported_semaphores;

  uint16_t _physical_dev_features;
  uint16_t _enabled_extensions;
  std::unique_ptr<vk_allocator> _allocator;
};

class vk_hardware_manager : public backend_hardware_manager {
public:
  vk_hardware_manager();

  std::size_t get_num_devices() const override;
  hardware_context *get_device(std::size_t index) override;
  device_id get_device_id(std::size_t index) const override;
  std::size_t get_num_platforms() const override;

  ~vk_hardware_manager();

private:
  vk::raii::Context _context;
  std::vector<vk_hardware_context> _devices;

  hardware_platform _hw_platform;
  vk::raii::Instance _instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT _debug_messenger = nullptr;

  static const std::vector<const char *> _validation_layers;
};

} // namespace rt
} // namespace hipsycl
