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

// This header describes the native objects that the Vulkan backend hands out
// for interoperability. It is included by user code through the SYCL interop
// API, and therefore deliberately only depends on the Vulkan C headers instead
// of the vulkan-hpp/vulkan-raii C++ bindings that the backend implementation
// uses internally.
//
// To avoid leaking vk backend implementation details to other parts of the
// codebase that could include this file, the interfaces below are declared
// which the runtime classes of the Vulkan backend implement.

#include <cstdint>

#include <vulkan/vulkan_core.h>

namespace hipsycl {
namespace rt {

// Native objects describing a SYCL device of the Vulkan backend.
struct vk_device_native_handle {
  // Instance the backend has created, corresponds to sycl::platform
  VkInstance instance;
  // Physical device, corresponds to sycl::device
  VkPhysicalDevice physical_device;
  // Logical device the backend has created for this physical device,
  // corresponds to sycl::context
  VkDevice device;
  // Index of the queue family that the backend submits work to
  uint32_t queue_family_index;
};

// Native objects describing a queue of the Vulkan backend. Note that the
// VkQueue is externally synchronized, and may only be used from within a
// custom operation.
struct vk_queue_native_handle {
  VkQueue queue;
  uint32_t queue_family_index;
  VkDevice device;
};

// Describes a point in the execution of a Vulkan queue using a timeline
// semaphore and the value it attains once the work has completed.
struct vk_event_native_handle {
  VkSemaphore semaphore;
  uint64_t value;
};

// Describes the Vulkan buffer backing a USM allocation. Device USM pointers of
// the Vulkan backend are device addresses obtained from
// VK_KHR_buffer_device_address, and as such always refer to a region of a
// VkBuffer.
struct vk_allocation_native_handle {
  VkBuffer buffer;
  // Offset of the queried pointer inside the buffer, in bytes
  VkDeviceSize offset;
  // Number of bytes between the queried pointer and the end of the allocation
  VkDeviceSize size;
};

// The kind of operating system handle used to share a VkDeviceMemory or a
// VkSemaphore payload between two Vulkan devices.
enum class vk_external_handle_type {
  none = 0,
  // A POSIX file descriptor, corresponding to
  // VK_EXTERNAL_{MEMORY,SEMAPHORE}_HANDLE_TYPE_OPAQUE_FD_BIT.
  opaque_fd = 1,
  // A Windows NT handle, corresponding to
  // VK_EXTERNAL_{MEMORY,SEMAPHORE}_HANDLE_TYPE_OPAQUE_WIN32_BIT.
  opaque_win32 = 2
};

// The handle type that is natively supported by the platform AdaptiveCpp has
// been built for. There is no implementation that supports both, so this is
// the handle type that applications will want to use.
#ifdef _WIN32
inline constexpr vk_external_handle_type vk_native_external_handle_type =
    vk_external_handle_type::opaque_win32;
#else
inline constexpr vk_external_handle_type vk_native_external_handle_type =
    vk_external_handle_type::opaque_fd;
#endif

// An operating system handle referring to a VkDeviceMemory or a VkSemaphore
// payload. Only the member matching `type` is meaningful.
//
// Ownership rules follow those of the underlying Vulkan extensions and differ
// between the two handle types:
//  * opaque_fd: exporting transfers ownership of the file descriptor to the
//    caller; importing transfers ownership to the Vulkan implementation. The
//    file descriptor must be close()d if it is not imported.
//  * opaque_win32: exporting creates a new NT handle owned by the caller;
//    importing does *not* transfer ownership. The handle must always be
//    released with CloseHandle(), also after a successful import.
struct vk_external_handle {
  vk_external_handle_type type = vk_external_handle_type::none;
  // Valid if type == opaque_fd
  int fd = -1;
  // Valid if type == opaque_win32. This is a Windows HANDLE; it is typed as
  // void* so that this header remains usable without <windows.h>.
  void *win32_handle = nullptr;

  bool is_valid() const {
    if (type == vk_external_handle_type::opaque_fd)
      return fd >= 0;
    if (type == vk_external_handle_type::opaque_win32)
      return win32_handle != nullptr;
    return false;
  }

  static vk_external_handle from_fd(int fd) {
    vk_external_handle h;
    h.type = vk_external_handle_type::opaque_fd;
    h.fd = fd;
    return h;
  }

  static vk_external_handle from_win32_handle(void *handle) {
    vk_external_handle h;
    h.type = vk_external_handle_type::opaque_win32;
    h.win32_handle = handle;
    return h;
  }
};

// Describes a USM allocation that has been exported as an opaque external
// handle, so that it can be imported into a VkDevice that AdaptiveCpp does not
// own. The importing application creates its own VkBuffer, allocates memory of
// allocation_size bytes from memory_type_index while chaining a
// VkImportMemoryFdInfoKHR or VkImportMemoryWin32HandleInfoKHR referring to the
// handle, and binds the buffer to it.
struct vk_external_memory_native_handle {
  // Handle referring to the VkDeviceMemory backing the allocation. See
  // vk_external_handle for the ownership rules.
  vk_external_handle handle;
  // Size of the VkDeviceMemory the handle refers to
  VkDeviceSize allocation_size;
  // Index of the memory type the allocation was made from. Only meaningful on
  // the same VkPhysicalDevice.
  uint32_t memory_type_index;
  // Offset of the queried pointer inside the allocation, in bytes
  VkDeviceSize offset;
  // Number of bytes between the queried pointer and the end of the allocation
  VkDeviceSize size;
};

// Describes a timeline semaphore that has been exported as an opaque external
// handle, together with the timeline value of interest.
struct vk_external_semaphore_native_handle {
  // Handle referring to the semaphore payload. See vk_external_handle for the
  // ownership rules.
  vk_external_handle handle;
  // Value the imported timeline attains once the work has completed
  uint64_t value;
};

// Implemented by vk_hardware_context
class vk_native_device_provider {
public:
  virtual vk_device_native_handle get_native_device_handle() const = 0;
  virtual ~vk_native_device_provider() {}
};

// Implemented by vk_hardware_context. Provides the external semaphore
// operations of the VkDevice that AdaptiveCpp has created, which allow
// synchronizing with an application that drives a VkDevice of its own.
class vk_native_external_semaphore_provider {
public:
  // Whether the external semaphore extension providing the given handle type
  // could be enabled on the device
  virtual bool
  has_external_semaphore_support(vk_external_handle_type type) const = 0;

  // Exports the given semaphore, which must belong to this device, as an
  // opaque handle of the requested type. Returns false on failure.
  virtual bool export_semaphore_handle(VkSemaphore semaphore,
                                       vk_external_handle_type type,
                                       vk_external_handle &out) const = 0;

  // Creates a semaphore on this device whose payload is imported from an
  // opaque handle of a timeline semaphore. Ownership of the handle follows the
  // rules described on vk_external_handle. Returns VK_NULL_HANDLE on failure.
  // The returned semaphore is owned by the runtime and must be released with
  // destroy_imported_semaphore().
  virtual VkSemaphore
  import_timeline_semaphore(const vk_external_handle &handle) = 0;

  virtual void destroy_imported_semaphore(VkSemaphore semaphore) = 0;

  virtual ~vk_native_external_semaphore_provider() {}
};

// Implemented by vk_allocator
class vk_native_external_memory_provider {
public:
  // Whether the external memory extension providing the given handle type
  // could be enabled on the device
  virtual bool
  has_external_memory_support(vk_external_handle_type type) const = 0;

  // Exports the allocation that ptr belongs to as an opaque handle of the
  // requested type. Returns false if ptr does not belong to an exportable
  // allocation of this allocator, or if the export failed.
  virtual bool
  export_allocation_handle(const void *ptr, vk_external_handle_type type,
                           vk_external_memory_native_handle &out) const = 0;

  virtual ~vk_native_external_memory_provider() {}
};

// Implemented by vk_allocator
class vk_native_allocation_provider {
public:
  // Returns false if ptr does not belong to an allocation of this allocator.
  virtual bool
  get_native_allocation(const void *ptr,
                        vk_allocation_native_handle &out) const = 0;
  virtual ~vk_native_allocation_provider() {}
};

// Implemented by vk_node_event and vk_imported_node_event
class vk_native_event_provider {
public:
  virtual vk_event_native_handle get_native_event_handle() const = 0;
  virtual ~vk_native_event_provider() {}
};

} // namespace rt
} // namespace hipsycl
