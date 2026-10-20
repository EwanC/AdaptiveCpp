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


#include "sycl_test_suite.hpp"
#include <boost/test/unit_test_suite.hpp>

BOOST_FIXTURE_TEST_SUITE(interop_handle_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(interop_handle_api) {
  namespace s = sycl;
  using namespace hipsycl;

  rt::device_id assigned_device{rt::backend_descriptor{rt::hardware_platform::cpu,
                                rt::api_platform::omp}, 0};
  s::queue q{s::device{assigned_device}};

  q.submit([&](s::handler &cgh) {
    cgh.AdaptiveCpp_enqueue_custom_operation([=](s::interop_handle &ih) {
      s::backend b = ih.get_backend();
      BOOST_CHECK(b == s::backend::omp);
    });
  });
  q.wait_and_throw();
}

#if defined(ACPP_GLUE_VK_INTEROP_AVAILABLE)
BOOST_AUTO_TEST_CASE(interop_handle_vk_native_queue) {
  namespace s = sycl;

  s::queue q;
  if (q.get_device().get_backend() != s::backend::vk) {
    BOOST_TEST_MESSAGE(
        "Skipping interop_handle_vk_native_queue: not a Vulkan device");
    return;
  }

  s::context ctx = q.get_context();
  if (ctx.get_devices().size() > 1) {
    BOOST_TEST_MESSAGE("Skipping interop_handle_vk_native_queue: context "
                       "contains more than 1 device");
    return;
  }

  VkDevice context_device = s::get_native<s::backend::vk>(q.get_context());
  BOOST_CHECK(context_device != VK_NULL_HANDLE);

  q.submit([&](s::handler &cgh) {
    cgh.AdaptiveCpp_enqueue_custom_operation([=](s::interop_handle &ih) {
      BOOST_CHECK(ih.get_backend() == s::backend::vk);

      auto native_queue = ih.get_native_queue<s::backend::vk>();
      BOOST_CHECK(native_queue.queue != VK_NULL_HANDLE);
      BOOST_CHECK(native_queue.device == context_device);
    });
  });
  q.wait_and_throw();
}
#endif

BOOST_AUTO_TEST_SUITE_END()
