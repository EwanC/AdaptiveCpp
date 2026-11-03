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
#include "hipSYCL/runtime/vk/vk_profiling.hpp"

namespace hipsycl {
namespace rt {

calibrated_timestamps vk_get_calibrated_timestamps(const vk::raii::Device &dev,
                                                   bool use_khr) {
  std::array<vk::CalibratedTimestampInfoEXT, 2> timestamp_infos = {
      vk::CalibratedTimestampInfoEXT(vk::TimeDomainEXT::eClockMonotonic),
      vk::CalibratedTimestampInfoEXT(vk::TimeDomainEXT::eDevice)};

  auto time_domains = use_khr ? dev.getCalibratedTimestampsKHR(timestamp_infos)
                              : dev.getCalibratedTimestampsEXT(timestamp_infos);
  auto &timestamps = time_domains.first; // Drop max deviation
  return calibrated_timestamps{timestamps[0], timestamps[1]};
}

uint64_t vk_get_query_pool_result(const vk::raii::QueryPool &query_pool,
                                  uint32_t query_index) {
  // Get timestamp data as 64-bit int rather than 32-bit default.
  auto [result, data] = query_pool.getResults<uint64_t>(
      query_index, 1, sizeof(uint64_t), 0,
      vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

  switch (result) {
  case vk::Result::eSuccess:
    break;
  case vk::Result::eNotReady:
    print_error(__acpp_here(),
                error_info{"vkGetQueryPoolResults returned VK_NOT_READY"});
    break;
  default:
    print_error(
        __acpp_here(),
        error_info{"vkGetQueryPoolResults returned an unexpected error"});
    break;
  }

  return data[0];
}

vk::raii::QueryPool vk_create_query_pool(const vk::raii::Device &dev,
                                         uint32_t query_count) {
  vk::QueryPoolCreateInfo query_pool_info{};
  query_pool_info.queryType = vk::QueryType::eTimestamp;
  query_pool_info.queryCount = query_count;
  return dev.createQueryPool(query_pool_info);
}

vk_query_slot::~vk_query_slot() { _allocator->release(_pool, _index); }

std::unique_ptr<vk_query_slot> vk_query_pool_allocator::allocate() {
  std::lock_guard<std::mutex> lock{_mutex};
  if (_free.empty()) {
    // All queries are still referenced by in-flight command-buffers or live
    // events, so grow instead of reusing one.
    _pools.push_back(vk_create_query_pool(_dev, queries_per_pool));
    const vk::raii::QueryPool *pool = &_pools.back();
    for (uint32_t i = queries_per_pool; i > 0; --i) {
      _free.emplace_back(pool, i - 1);
    }
    HIPSYCL_DEBUG_INFO << "vk_query_pool_allocator: allocated query pool #"
                       << _pools.size() << std::endl;
  }
  auto [pool, index] = _free.back();
  _free.pop_back();
  return std::make_unique<vk_query_slot>(shared_from_this(), pool, index);
}

void vk_query_pool_allocator::release(const vk::raii::QueryPool *pool,
                                      uint32_t index) {
  std::lock_guard<std::mutex> lock{_mutex};
  _free.emplace_back(pool, index);
}

} // namespace rt
} // namespace hipsycl
