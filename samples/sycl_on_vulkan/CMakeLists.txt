find_package(Vulkan)
find_package(glfw3 CONFIG QUIET)

if(NOT Vulkan_FOUND OR NOT glfw3_FOUND)
  message(WARNING
    "sycl_on_vulkan: skipping sample, it requires the Vulkan SDK and GLFW3 "
    "(Vulkan_FOUND=${Vulkan_FOUND}, glfw3_FOUND=${glfw3_FOUND}).")
  return()
endif()

add_executable(sycl_on_vulkan sycl_on_vulkan.cpp)
target_link_libraries(sycl_on_vulkan PRIVATE Vulkan::Vulkan glfw)
add_sycl_to_target(TARGET sycl_on_vulkan SOURCES sycl_on_vulkan.cpp)

install(TARGETS sycl_on_vulkan
        RUNTIME DESTINATION share/AdaptiveCpp/samples/)
