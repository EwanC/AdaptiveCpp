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

#include "../instrumentation.hpp"
#include "hipSYCL/runtime/error.hpp"
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

namespace hipsycl {
namespace rt {

class vk_hardware_context;

// Helper function for vkGetCalibratedTimestampsKHR() with the timestamps
// needed for profiling
struct calibrated_timestamps {
  uint64_t clock_monotonic;
  uint64_t device;
};
calibrated_timestamps vk_get_calibrated_timestamps(const vk::raii::Device &,
                                                   bool);

// Helper function for calling vkGetQueryPoolResults on a single query
uint64_t vk_get_query_pool_result(const vk::raii::QueryPool &,
                                  uint32_t query_index);

// Helper function for creating a timestamp query pool
vk::raii::QueryPool vk_create_query_pool(const vk::raii::Device &,
                                         uint32_t query_count);

class vk_query_pool_allocator;

// A single timestamp query handed out by vk_query_pool_allocator. The query
// is returned to the allocator when this object is destroyed, so it must be
// kept alive until both the command-buffer writing it has completed and
// nobody needs to read the result anymore.
class vk_query_slot {
public:
  vk_query_slot(std::shared_ptr<vk_query_pool_allocator> allocator,
                const vk::raii::QueryPool *pool, uint32_t index)
      : _allocator{std::move(allocator)}, _pool{pool}, _index{index} {}
  ~vk_query_slot();

  vk_query_slot(const vk_query_slot &) = delete;
  vk_query_slot &operator=(const vk_query_slot &) = delete;

  vk::QueryPool get_pool() const { return **_pool; }
  uint32_t get_index() const { return _index; }
  uint64_t get_result() const {
    return vk_get_query_pool_result(*_pool, _index);
  }

private:
  std::shared_ptr<vk_query_pool_allocator> _allocator;
  const vk::raii::QueryPool *_pool;
  uint32_t _index;
};

// Hands out timestamp queries from a growing set of query pools, so that
// profiled operations do not each need to create their own query pool.
// Thread-safe; queries may be released from any thread.
class vk_query_pool_allocator
    : public std::enable_shared_from_this<vk_query_pool_allocator> {
public:
  explicit vk_query_pool_allocator(const vk::raii::Device &dev) : _dev{dev} {}

  std::unique_ptr<vk_query_slot> allocate();

private:
  friend class vk_query_slot;
  void release(const vk::raii::QueryPool *pool, uint32_t index);

  static constexpr uint32_t queries_per_pool = 256;

  const vk::raii::Device &_dev;
  std::mutex _mutex;
  // deque keeps references to existing pools stable when growing
  std::deque<vk::raii::QueryPool> _pools;
  std::vector<std::pair<const vk::raii::QueryPool *, uint32_t>> _free;
};

// Converts a device timestamp (in device ticks, as returned by
// vkGetQueryPoolResults) to host CLOCK_MONOTONIC nanoseconds, using a
// calibrated pair of reference timestamps taken at the same instant.
//
// `ns_per_tick` is VkPhysicalDeviceLimits::timestampPeriod and `valid_bits`
// is VkQueueFamilyProperties::timestampValidBits. Only the low `valid_bits`
// of device timestamps are meaningful, so the tick difference is computed
// modulo 2^valid_bits and interpreted as signed, which handles wrap-around
// between the reference and the query.
inline int64_t vk_device_to_host_time(uint64_t host_ref_ns,
                                      uint64_t device_ref_ticks,
                                      uint64_t device_ticks, double ns_per_tick,
                                      uint32_t valid_bits) {
  uint64_t delta = device_ticks - device_ref_ticks;
  int64_t signed_delta;
  if (valid_bits == 0 || valid_bits >= 64) {
    signed_delta = static_cast<int64_t>(delta);
  } else {
    const uint64_t mask = (uint64_t{1} << valid_bits) - 1;
    const uint64_t sign_bit = uint64_t{1} << (valid_bits - 1);
    delta &= mask;
    signed_delta = (delta & sign_bit) ? static_cast<int64_t>(delta | ~mask)
                                      : static_cast<int64_t>(delta);
  }

  return static_cast<int64_t>(host_ref_ns) +
         std::llround(static_cast<double>(signed_delta) * ns_per_tick);
}

// Synchronous timestamp used for submission metric
class vk_sync_timestamp : public instrumentations::submission_timestamp {
public:
  vk_sync_timestamp(const vk::raii::Device &dev, bool use_khr) {
    // Use VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR value which is already in the
    // unit of nanoseconds
    uint64_t now = vk_get_calibrated_timestamps(dev, use_khr).clock_monotonic;
    _time = profiler_clock::time_point{profiler_clock::duration{now}};
  }

  profiler_clock::time_point get_time_point() const override { return _time; }

  void wait() const override {}

private:
  profiler_clock::time_point _time;
};

// Asynchronous timestamp used for command start/finish metric
// This needs to support device (via vkCommandBuffer) or
// host (via host_worker thread)  async execution.
template <class Base> class vk_async_timestamp : public Base {
public:
  // `ref_timestamps` is a calibrated (host, device) timestamp pair. Start and
  // finish timestamps of the same operation should share the same pair, so
  // that calibration jitter cannot make the measured duration negative.
  vk_async_timestamp(const vk::raii::Device &dev,
                     vk_query_pool_allocator &query_allocator,
                     vk::Semaphore sem, bool use_khr,
                     calibrated_timestamps ref_timestamps, double ns_per_tick,
                     uint32_t valid_bits)
      : _dev(dev), _use_khr(use_khr), _ref_timestamps(ref_timestamps),
        _ns_per_tick(ns_per_tick), _valid_bits(valid_bits), _sem(sem),
        _sem_wait_val(0), _host_timestamp(0) {
    // Query owned by this instrumentation object
    _query = query_allocator.allocate();
  }

  // Getters allow vkQueue to append command-buffer commands using this query
  vk::QueryPool get_query_pool() { return _query->get_pool(); }
  uint32_t get_query_index() const { return _query->get_index(); }

  // Set the value of the timeline semaphore to wait on for command completion
  void set_semaphore_wait_val(uint64_t wait_val) { _sem_wait_val = wait_val; }

  // Called as part of asynchronous host execution.
  void take_host_timestamp() {
    _host_timestamp =
        vk_get_calibrated_timestamps(_dev, _use_khr).clock_monotonic;
  }

  profiler_clock::time_point get_time_point() const override {
    // If command was executed on host, directly convert to time point
    if (_host_timestamp != 0) {
      return profiler_clock::time_point{
          profiler_clock::duration{_host_timestamp}};
    }

    // If command was executed on device, convert the device timestamp
    // (in device ticks) to host time using the calibrated reference pair.
    // Get captured device timestamp from query pool in domain
    // VK_TIME_DOMAIN_DEVICE_KHR
    // The result is cached so repeated queries do not hit the driver.
    std::call_once(_device_time_once,
                   [this] { _device_time = _query->get_result(); });
    const uint64_t device_time = _device_time;

    const int64_t host_time = vk_device_to_host_time(
        _ref_timestamps.clock_monotonic, _ref_timestamps.device, device_time,
        _ns_per_tick, _valid_bits);
    return profiler_clock::time_point{profiler_clock::duration{host_time}};
  }

  void wait() const override {
    HIPSYCL_DEBUG_INFO << "vk_async_timestamp: semaphore " << _sem
                       << " wait on " << _sem_wait_val << std::endl;

    // Wait on completion of semaphore indicating async work has completed
    vk::SemaphoreWaitInfo wait_info({}, 1, &_sem, &_sem_wait_val);
    vk::Result wait_ret_code;
    do {
      wait_ret_code = _dev.waitSemaphores(wait_info, UINT64_MAX);
    } while (vk::Result::eTimeout == wait_ret_code);

    if (wait_ret_code != vk::Result::eSuccess) {
      std::string err_msg("Semaphore wait failed with unexpected return code ");
      err_msg += std::to_string(static_cast<VkResult>(wait_ret_code));
      print_error(__acpp_here(), error_info{err_msg});
    }
  }

private:
  // Members below used for device execution
  const vk::raii::Device &_dev;
  bool _use_khr;
  std::unique_ptr<vk_query_slot> _query;
  mutable std::once_flag _device_time_once;
  mutable uint64_t _device_time = 0;
  calibrated_timestamps _ref_timestamps;
  double _ns_per_tick;
  uint32_t _valid_bits;

  // Members used to implement wait() method
  vk::Semaphore _sem;
  uint64_t _sem_wait_val;

  // Members below used for host execution
  uint64_t _host_timestamp;
};

using vk_execution_start_timestamp =
    vk_async_timestamp<instrumentations::execution_start_timestamp>;
using vk_execution_finish_timestamp =
    vk_async_timestamp<instrumentations::execution_finish_timestamp>;

struct vk_async_profiling {
  std::shared_ptr<vk_execution_start_timestamp> start_time;
  std::shared_ptr<vk_execution_finish_timestamp> finish_time;
};

} // namespace rt
} // namespace hipsycl
