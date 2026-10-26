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

#ifdef SYCL_EXT_ACPP_BACKEND_VK
// Interop with the Vulkan backend requires the Vulkan headers to be available
// to user code. AdaptiveCpp does not add them to the include path, so only
// enable the interop API if they can be found.
#if defined(__has_include)
#if __has_include(<vulkan/vulkan_core.h>)
#define ACPP_GLUE_VK_INTEROP_AVAILABLE
// Feature test macro of the sycl::AdaptiveCpp_vk_interop extension, which is
// only defined here as it also depends on the Vulkan headers
#define ACPP_EXT_VK_INTEROP
#endif
#endif
#endif

#ifdef ACPP_GLUE_VK_INTEROP_AVAILABLE

#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/vk/vk_native_objects.hpp"

#include "hipSYCL/sycl/context.hpp"
#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/sycl/event.hpp"
#include "hipSYCL/sycl/platform.hpp"

#include <optional>

#ifdef _WIN32
// Declared manually so that this header does not have to pull in the entirety
// of <windows.h>. This matches the declaration in <windows.h>, so including
// that in addition is fine.
extern "C" __declspec(dllimport) int __stdcall CloseHandle(void *hObject);
#else
#include <unistd.h>
#endif

namespace hipsycl {
namespace glue {

namespace vk_interop_detail {

// Returns the native handles of the single Vulkan device contained in ctx.
// Interop requires an unambiguous VkDevice, so contexts containing more than
// one Vulkan device are rejected.
inline bool get_unique_device_handle(const sycl::context &ctx,
                                     rt::vk_device_native_handle &out,
                                     rt::device_id &out_dev) {
  rt::backend *b =
      ctx.AdaptiveCpp_runtime()->backends().get(rt::backend_id::vk);
  if (!b) {
    return false;
  }

  bool found = false;
  for (const sycl::device &dev : ctx.get_devices()) {
    rt::device_id dev_id = sycl::detail::extract_rt_device(dev);
    if (dev_id.get_backend() != rt::backend_id::vk) {
      continue;
    }

    if (found) {
      // More than one Vulkan device in this context
      return false;
    }

    auto *provider = dynamic_cast<rt::vk_native_device_provider *>(
        b->get_hardware_manager()->get_device(dev_id.get_id()));
    if (!provider) {
      return false;
    }

    out = provider->get_native_device_handle();
    out_dev = dev_id;
    found = true;
  }

  return found;
}

inline bool get_device_handle(const sycl::device &d,
                              rt::vk_device_native_handle &out) {
  rt::device_id dev_id = sycl::detail::extract_rt_device(d);

  if (dev_id.get_backend() != rt::backend_id::vk) {
    rt::register_error(
        __acpp_here(),
        rt::error_info{
            "vk interop: device does not belong to the Vulkan backend",
            rt::error_type::invalid_parameter_error});
    return false;
  }

  rt::backend *b = d.AdaptiveCpp_runtime()->backends().get(rt::backend_id::vk);
  if (!b) {
    rt::register_error(
        __acpp_here(),
        rt::error_info{"vk interop: Vulkan backend not available",
                       rt::error_type::runtime_error});
    return false;
  }

  auto *provider = dynamic_cast<rt::vk_native_device_provider *>(
      b->get_hardware_manager()->get_device(dev_id.get_id()));
  if (!provider) {
    rt::register_error(
        __acpp_here(),
        rt::error_info{
            "vk interop: device does not expose native Vulkan objects",
            rt::error_type::runtime_error});
    return false;
  }

  out = provider->get_native_device_handle();
  return true;
}

// Returns the external semaphore operations of a device of the Vulkan backend,
// or nullptr if dev is not such a device.
inline rt::vk_native_external_semaphore_provider *
get_external_semaphore_provider(const sycl::device &dev) {
  rt::device_id dev_id = sycl::detail::extract_rt_device(dev);
  if (dev_id.get_backend() != rt::backend_id::vk) {
    return nullptr;
  }

  rt::backend *b =
      dev.AdaptiveCpp_runtime()->backends().get(rt::backend_id::vk);
  if (!b) {
    return nullptr;
  }

  return dynamic_cast<rt::vk_native_external_semaphore_provider *>(
      b->get_hardware_manager()->get_device(dev_id.get_id()));
}

// Returns the external semaphore operations of the device of the Vulkan
// backend that has created the given VkDevice, or nullptr if there is none.
inline rt::vk_native_external_semaphore_provider *
get_external_semaphore_provider(rt::runtime *runtime, VkDevice device) {
  rt::backend *b = runtime->backends().get(rt::backend_id::vk);
  if (!b || device == VK_NULL_HANDLE) {
    return nullptr;
  }

  rt::backend_hardware_manager *hw_manager = b->get_hardware_manager();
  for (std::size_t i = 0; i < hw_manager->get_num_devices(); ++i) {
    auto *dev_provider = dynamic_cast<rt::vk_native_device_provider *>(
        hw_manager->get_device(i));
    if (dev_provider &&
        dev_provider->get_native_device_handle().device == device) {
      return dynamic_cast<rt::vk_native_external_semaphore_provider *>(
          hw_manager->get_device(i));
    }
  }

  return nullptr;
}

// Turns an event that the backend has created from a native Vulkan event of
// the given device into a SYCL event.
inline sycl::event
make_imported_event(rt::runtime *runtime, rt::device_id dev,
                    std::shared_ptr<rt::dag_node_event> runtime_event) {
  auto op = std::make_unique<rt::kernel_operation>(
      "<native Vulkan semaphore>", rt::kernel_launcher({}, {}),
      rt::requirements_list{runtime});
  rt::dag_node_ptr node = std::make_shared<rt::dag_node>(
      rt::execution_hints{}, rt::node_list_t{}, std::move(op), runtime);
  node->assign_to_device(dev);
  node->assign_to_executor(nullptr);
  node->assign_to_execution_lane(nullptr);
  node->assign_execution_index(0);
  node->get_execution_hints().set_hint(rt::hints::instant_execution{});
  node->get_operation()->get_instrumentations().mark_set_complete();
  node->mark_submitted(std::move(runtime_event));

  return sycl::event{node};
}

inline rt::vk_native_external_memory_provider *
get_external_memory_provider(const sycl::device &dev) {
  rt::device_id dev_id = sycl::detail::extract_rt_device(dev);
  if (dev_id.get_backend() != rt::backend_id::vk) {
    return nullptr;
  }

  rt::backend *b =
      dev.AdaptiveCpp_runtime()->backends().get(rt::backend_id::vk);
  if (!b) {
    return nullptr;
  }

  return dynamic_cast<rt::vk_native_external_memory_provider *>(
      b->get_allocator(dev_id));
}

} // namespace vk_interop_detail

template <> struct backend_interop<sycl::backend::vk> {

  using error_type = int;

  using native_mem_type = void *;
  using native_platform_type = VkInstance;
  using native_device_type = VkPhysicalDevice;
  using native_context_type = VkDevice;
  using native_queue_type = rt::vk_queue_native_handle;
  using native_event_type = rt::vk_event_native_handle;
  using native_allocation_type = rt::vk_allocation_native_handle;

  template <class Accessor_type>
  static native_mem_type get_native_mem(const Accessor_type &a) {
    return a.get_pointer();
  }

  static native_platform_type get_native_platform(const sycl::platform &p) {
    if (p.get_backend() != sycl::backend::vk) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_platform: platform does not belong to "
                         "the Vulkan backend",
                         rt::error_type::invalid_parameter_error});
      return VK_NULL_HANDLE;
    }

    std::vector<sycl::device> devices = p.get_devices();
    if (devices.empty()) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{
              "get_native_platform: platform does not contain any devices",
              rt::error_type::runtime_error});
      return VK_NULL_HANDLE;
    }

    // All Vulkan devices of the backend originate from the same instance
    rt::vk_device_native_handle handle;
    if (!vk_interop_detail::get_device_handle(devices.front(), handle)) {
      return VK_NULL_HANDLE;
    }

    return handle.instance;
  }

  static native_device_type get_native_device(const sycl::device &d) {
    rt::vk_device_native_handle handle;
    if (!vk_interop_detail::get_device_handle(d, handle)) {
      return VK_NULL_HANDLE;
    }

    return handle.physical_device;
  }

  static native_context_type get_native_context(const sycl::context &ctx) {
    rt::vk_device_native_handle handle;
    rt::device_id dev_id;
    if (!vk_interop_detail::get_unique_device_handle(ctx, handle, dev_id)) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_context: context does not contain exactly "
                         "one Vulkan device",
                         rt::error_type::invalid_parameter_error});
      return VK_NULL_HANDLE;
    }

    return handle.device;
  }

  static native_queue_type get_native_queue(void *launcher_params) {
    const native_queue_type invalid_queue{VK_NULL_HANDLE, 0, VK_NULL_HANDLE};

    if (!launcher_params) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_queue: invalid (null) launcher params",
                         rt::error_type::invalid_parameter_error});
      return invalid_queue;
    }

    void *native_type =
        static_cast<rt::inorder_queue *>(launcher_params)->get_native_type();
    if (!native_type) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{
              "get_native_queue: queue does not expose a native Vulkan queue",
              rt::error_type::runtime_error});
      return invalid_queue;
    }

    return *static_cast<native_queue_type *>(native_type);
  }

  static native_event_type get_native_event(const sycl::event &e) {
    const native_event_type invalid_event{VK_NULL_HANDLE, 0};

    rt::dag_node_ptr node = sycl::detail::extract_rt_event(e);
    if (!node) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: invalid event",
                         rt::error_type::invalid_parameter_error});
      return invalid_event;
    }

    // The native synchronization point only exists once the operation has
    // been submitted to the backend.
    if (!node->is_submitted()) {
      rt::runtime_keep_alive_token{}.get()->dag().flush_and_gc();
    }

    if (!node->is_submitted()) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: event has not been submitted",
                         rt::error_type::runtime_error});
      return invalid_event;
    }

    if (node->is_cancelled()) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: event has been cancelled",
                         rt::error_type::runtime_error});
      return invalid_event;
    }

    if (node->get_assigned_device().get_backend() != rt::backend_id::vk) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: event does not belong to the "
                         "Vulkan backend",
                         rt::error_type::invalid_parameter_error});
      return invalid_event;
    }

    auto event = node->get_event();
    if (!event) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: event has not been submitted",
                         rt::error_type::runtime_error});
      return invalid_event;
    }

    auto *vk_event = dynamic_cast<rt::vk_native_event_provider *>(event.get());
    if (!vk_event) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: event is not backed by a native "
                         "Vulkan timeline semaphore",
                         rt::error_type::feature_not_supported});
      return invalid_event;
    }

    return vk_event->get_native_event_handle();
  }

  static sycl::event make_sycl_event(const native_event_type &native_event,
                                     const sycl::context &ctx) {
    if (native_event.semaphore == VK_NULL_HANDLE) {
      throw sycl::exception{sycl::make_error_code(sycl::errc::invalid),
                            "make_event: invalid native Vulkan semaphore"};
    }

    rt::runtime *runtime = ctx.AdaptiveCpp_runtime();
    rt::backend *backend = runtime->backends().get(rt::backend_id::vk);
    if (!backend) {
      throw sycl::exception{sycl::make_error_code(sycl::errc::backend_mismatch),
                            "make_event: Vulkan backend not available"};
    }

    std::vector<rt::device_id> vk_devices;
    for (const sycl::device &dev : ctx.get_devices()) {
      rt::device_id dev_id = sycl::detail::extract_rt_device(dev);
      if (dev_id.get_backend() == rt::backend_id::vk) {
        vk_devices.push_back(dev_id);
      }
    }

    if (vk_devices.empty()) {
      throw sycl::exception{
          sycl::make_error_code(sycl::errc::backend_mismatch),
          "make_event: the supplied context does not contain a Vulkan device"};
    }

    if (native_event.device == VK_NULL_HANDLE && vk_devices.size() > 1) {
      throw sycl::exception{
          sycl::make_error_code(sycl::errc::invalid),
          "make_event: the VkDevice of the native Vulkan event must be set, as "
          "the supplied context contains more than one Vulkan device"};
    }

    // The backend only accepts the native event for the device it belongs to
    for (rt::device_id dev : vk_devices) {
      std::shared_ptr<rt::dag_node_event> runtime_event =
          backend->create_event_from_native_handle(&native_event, dev);
      if (runtime_event) {
        return vk_interop_detail::make_imported_event(runtime, dev,
                                                      std::move(runtime_event));
      }
    }

    throw sycl::exception{
        sycl::make_error_code(sycl::errc::invalid),
        "make_event: the native Vulkan event does not belong to a device of "
        "the supplied context"};
  }

  static native_allocation_type
  get_native_allocation(const void *ptr, const sycl::context &ctx) {
    const native_allocation_type invalid_allocation{VK_NULL_HANDLE, 0, 0};

    rt::backend *b =
        ctx.AdaptiveCpp_runtime()->backends().get(rt::backend_id::vk);
    if (!b) {
      return invalid_allocation;
    }

    for (const sycl::device &dev : ctx.get_devices()) {
      rt::device_id dev_id = sycl::detail::extract_rt_device(dev);
      if (dev_id.get_backend() != rt::backend_id::vk) {
        continue;
      }

      auto *provider = dynamic_cast<rt::vk_native_allocation_provider *>(
          b->get_allocator(dev_id));
      if (!provider) {
        continue;
      }

      native_allocation_type result;
      if (provider->get_native_allocation(ptr, result)) {
        return result;
      }
    }

    return invalid_allocation;
  }

  static constexpr bool can_make_platform = false;
  static constexpr bool can_make_device = false;
  static constexpr bool can_make_context = false;
  static constexpr bool can_make_queue = false;
  static constexpr bool can_make_event = true;
  static constexpr bool can_make_buffer = false;
  static constexpr bool can_make_sampled_image = false;
  static constexpr bool can_make_image_sampler = false;
  static constexpr bool can_make_stream = false;
  static constexpr bool can_make_kernel = false;
  static constexpr bool can_make_module = false;

  static constexpr bool can_extract_native_platform = true;
  static constexpr bool can_extract_native_device = true;
  static constexpr bool can_extract_native_context = true;
  // sycl::get_native(queue) is not implemented, since AdaptiveCpp may use a
  // different backend queue for each operation. Use
  // interop_handle::get_native_queue() inside a custom operation instead.
  static constexpr bool can_extract_native_queue = false;
  static constexpr bool can_extract_native_event = true;
  static constexpr bool can_extract_native_buffer = false;
  static constexpr bool can_extract_native_sampled_image = false;
  static constexpr bool can_extract_native_image_sampler = false;
  static constexpr bool can_extract_native_stream = false;
  static constexpr bool can_extract_native_kernel = false;
  static constexpr bool can_extract_native_module = false;
  static constexpr bool can_extract_native_device_event = false;
  static constexpr bool can_extract_native_mem = true;
  static constexpr bool can_extract_native_allocation = true;
};

} // namespace glue
} // namespace hipsycl

namespace hipsycl {
namespace sycl {

// Interop with Vulkan devices that AdaptiveCpp does not own. An application
// that drives a VkDevice of its own - for example because it needs graphics,
// presentation, or extensions and features the backend does not enable - can
// keep its own VkInstance and VkDevice, and share memory and synchronization
// with the Vulkan backend through opaque external handles of
// VK_KHR_external_memory_{fd,win32} and VK_KHR_external_semaphore_{fd,win32}.
// Both devices must be created for the same physical device and driver, which
// is the case if VkPhysicalDeviceIDProperties reports the same deviceUUID and
// driverUUID for them. VkPhysicalDevice handles cannot be compared instead, as
// they differ between VkInstances, but the properties can be queried for the
// handle that sycl::get_native<sycl::backend::vk>(device) returns.
namespace AdaptiveCpp_vk_interop {

using external_handle_type = rt::vk_external_handle_type;
using external_handle = rt::vk_external_handle;
using external_memory_handle = rt::vk_external_memory_native_handle;
using external_semaphore_handle = rt::vk_external_semaphore_native_handle;

// The handle type supported by the platform AdaptiveCpp was built for. No
// Vulkan implementation supports both opaque file descriptors and Win32
// handles, so this is the handle type applications will want to use, and the
// default of all entry points below.
inline constexpr external_handle_type native_external_handle_type =
    rt::vk_native_external_handle_type;

// Closes an external handle that was returned by export_memory() or
// export_semaphore(). File descriptors must not be closed once they have been
// imported, since the import takes ownership of them. Win32 handles on the
// other hand must always be closed, also after a successful import.
inline void close_external_handle(const external_handle &handle) {
  if (!handle.is_valid()) {
    return;
  }
#ifdef _WIN32
  if (handle.type == external_handle_type::opaque_win32) {
    ::CloseHandle(handle.win32_handle);
  }
#else
  if (handle.type == external_handle_type::opaque_fd) {
    ::close(handle.fd);
  }
#endif
}

// Whether USM allocations of this device can be exported as opaque external
// handles of the given type.
inline bool has_external_memory_support(
    const device &dev,
    external_handle_type type = native_external_handle_type) {
  auto *provider = glue::vk_interop_detail::get_external_memory_provider(dev);
  return provider && provider->has_external_memory_support(type);
}

// Whether timeline semaphores of this device can be exported to, and imported
// from, opaque external handles of the given type.
inline bool has_external_semaphore_support(
    const device &dev,
    external_handle_type type = native_external_handle_type) {
  auto *provider =
      glue::vk_interop_detail::get_external_semaphore_provider(dev);
  return provider && provider->has_external_semaphore_support(type);
}

// Exports the memory of the USM allocation that ptr belongs to as an opaque
// external handle, which the application can import into its own VkDevice.
// The allocation must have been created with the
// sycl::property::usm::AdaptiveCpp_exportable property. On failure, a handle
// for which handle.is_valid() is false is returned. The returned handle is
// owned by the caller and has to be released with close_external_handle()
// once it is no longer needed - note that importing a file descriptor already
// transfers its ownership, while importing a Win32 handle does not.
inline external_memory_handle
export_memory(const void *ptr, const context &ctx,
              external_handle_type type = native_external_handle_type) {
  external_memory_handle result{external_handle{}, 0, 0, 0, 0};

  for (const device &dev : ctx.get_devices()) {
    auto *provider = glue::vk_interop_detail::get_external_memory_provider(dev);
    if (!provider) {
      continue;
    }

    if (provider->export_allocation_handle(ptr, type, result)) {
      return result;
    }
  }

  return external_memory_handle{external_handle{}, 0, 0, 0, 0};
}

// Exports the timeline semaphore backing a SYCL event as an opaque external
// handle, together with the value the timeline attains once the operation has
// completed. The application must not signal values on the imported payload,
// since the semaphore is the timeline AdaptiveCpp uses internally. On failure,
// a handle for which handle.is_valid() is false is returned. See
// export_memory() for the ownership of the returned handle.
inline external_semaphore_handle
export_semaphore(const event &e,
                 external_handle_type type = native_external_handle_type) {
  auto native_event = glue::backend_interop<backend::vk>::get_native_event(e);
  if (native_event.semaphore == VK_NULL_HANDLE) {
    return external_semaphore_handle{external_handle{}, 0};
  }

  rt::runtime_keep_alive_token runtime;
  auto *provider = glue::vk_interop_detail::get_external_semaphore_provider(
      runtime.get(), native_event.device);
  if (!provider) {
    rt::register_error(
        __acpp_here(),
        rt::error_info{"export_semaphore: event does not belong to a device "
                       "of the Vulkan backend",
                       rt::error_type::invalid_parameter_error});
    return external_semaphore_handle{external_handle{}, 0};
  }

  external_handle handle;
  if (!provider->export_semaphore_handle(native_event.semaphore, type,
                                         handle)) {
    return external_semaphore_handle{external_handle{}, 0};
  }

  return external_semaphore_handle{handle, native_event.value};
}

// A timeline semaphore of the application, imported into the VkDevice that
// AdaptiveCpp has created for a device. SYCL work can be made to wait on values
// of the imported timeline by turning them into events with make_event(). The
// object must not be destroyed or reset before all SYCL work that depends on
// these events has completed, as the semaphore is destroyed with it. It keeps
// the SYCL runtime alive until it is destroyed or reset.
class imported_semaphore {
public:
  imported_semaphore() = default;

  // Imports the timeline semaphore described by an opaque external handle
  // into the VkDevice of dev. If the handle is a file descriptor, ownership of
  // it is transferred to the Vulkan implementation on success. A Win32 handle
  // on the other hand stays owned by the caller, which has to close it once
  // the import has returned.
  imported_semaphore(const device &dev, const external_handle &handle) {
    auto *provider =
        glue::vk_interop_detail::get_external_semaphore_provider(dev);
    if (!provider) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"imported_semaphore: device does not belong to the "
                         "Vulkan backend",
                         rt::error_type::invalid_parameter_error});
      return;
    }

    _semaphore = provider->import_timeline_semaphore(handle);
    if (_semaphore != VK_NULL_HANDLE) {
      _provider = provider;
      _device.emplace(dev);
    }
  }

  imported_semaphore(const imported_semaphore &) = delete;
  imported_semaphore &operator=(const imported_semaphore &) = delete;

  imported_semaphore(imported_semaphore &&other) noexcept
      : _provider{other._provider}, _semaphore{other._semaphore},
        _device{std::move(other._device)} {
    other._provider = nullptr;
    other._semaphore = VK_NULL_HANDLE;
    other._device.reset();
  }

  imported_semaphore &operator=(imported_semaphore &&other) noexcept {
    if (this != &other) {
      reset();
      _provider = other._provider;
      _semaphore = other._semaphore;
      _device = std::move(other._device);
      other._provider = nullptr;
      other._semaphore = VK_NULL_HANDLE;
      other._device.reset();
    }
    return *this;
  }

  ~imported_semaphore() { reset(); }

  void reset() {
    if (_provider && _semaphore != VK_NULL_HANDLE) {
      _provider->destroy_imported_semaphore(_semaphore);
    }
    _provider = nullptr;
    _semaphore = VK_NULL_HANDLE;
    // Only release the runtime once the semaphore has been destroyed
    _device.reset();
  }

  // The semaphore created inside the VkDevice of get_device(). It shares its
  // payload with the semaphore the handle was exported from.
  VkSemaphore get() const { return _semaphore; }

  // The device the semaphore has been imported into
  device get_device() const {
    if (!*this) {
      throw exception{make_error_code(errc::invalid),
                      "imported_semaphore: no semaphore has been imported"};
    }
    return *_device;
  }

  explicit operator bool() const { return _semaphore != VK_NULL_HANDLE; }

private:
  rt::vk_native_external_semaphore_provider *_provider = nullptr;
  VkSemaphore _semaphore = VK_NULL_HANDLE;
  // Keeps the runtime, which owns the semaphore, alive
  std::optional<device> _device;
};

// Returns an event that is complete once the imported timeline semaphore
// reaches the given value. SYCL operations can be made to wait on the
// application's work by depending on this event.
inline event make_event(const imported_semaphore &semaphore, uint64_t value) {
  if (!semaphore) {
    throw exception{make_error_code(errc::invalid),
                    "make_event: invalid imported semaphore"};
  }

  const device dev = semaphore.get_device();
  rt::runtime *runtime = dev.AdaptiveCpp_runtime();
  rt::backend *backend = runtime->backends().get(rt::backend_id::vk);
  const rt::device_id dev_id = sycl::detail::extract_rt_device(dev);
  const rt::vk_event_native_handle native_event{semaphore.get(), value};

  std::shared_ptr<rt::dag_node_event> runtime_event =
      backend ? backend->create_event_from_native_handle(&native_event, dev_id)
              : nullptr;
  if (!runtime_event) {
    throw exception{make_error_code(errc::invalid),
                    "make_event: could not import the semaphore"};
  }

  return glue::vk_interop_detail::make_imported_event(runtime, dev_id,
                                                      std::move(runtime_event));
}
} // namespace AdaptiveCpp_vk_interop
} // namespace sycl
} // namespace hipsycl

#endif // ACPP_GLUE_VK_INTEROP_AVAILABLE
