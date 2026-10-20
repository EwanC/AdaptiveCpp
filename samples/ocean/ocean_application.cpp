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

// The Vulkan setup of this sample is derived from the Vulkan Tutorial
// (https://vulkan-tutorial.com), whose code samples are licensed as
// CC0 1.0 Universal, and from the ocean sample of the Khronos OpenCL-SDK.

#include "ocean_application.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

#include <glm/gtc/matrix_transform.hpp>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

namespace {

constexpr std::size_t max_frames_in_flight = 2;
constexpr VkFormat interop_texture_format = VK_FORMAT_R32G32B32A32_SFLOAT;

const char *const validation_layer_name = "VK_LAYER_KHRONOS_validation";

void check(VkResult result, const char *message) {
  if (result != VK_SUCCESS)
    throw std::runtime_error{std::string{message} + " (VkResult " +
                             std::to_string(static_cast<int>(result)) + ")"};
}

std::string executable_directory() {
  std::string path;
#if defined(_WIN32)
  char buffer[MAX_PATH] = {};
  const DWORD length = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
  path.assign(buffer, length);
#elif defined(__APPLE__)
  char buffer[4096] = {};
  std::uint32_t size = sizeof(buffer);
  if (_NSGetExecutablePath(buffer, &size) == 0)
    path.assign(buffer);
#else
  char buffer[4096] = {};
  const ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (length > 0)
    path.assign(buffer, static_cast<std::size_t>(length));
#endif

  const std::size_t separator = path.find_last_of("/\\");
  if (separator == std::string::npos)
    return ".";
  return path.substr(0, separator);
}

std::vector<char> read_binary_file(const std::string &filename) {
  std::ifstream file{filename, std::ios::ate | std::ios::binary};
  if (!file.is_open())
    throw std::runtime_error{"Could not open file: " + filename};

  const std::size_t size = static_cast<std::size_t>(file.tellg());
  std::vector<char> buffer(size);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(size));
  return buffer;
}

std::vector<char> read_shader(const char *filename) {
  return read_binary_file(executable_directory() + "/shaders/" + filename);
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT,
               VkDebugUtilsMessageTypeFlagsEXT,
               const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
               void *) {
  std::cerr << "[vulkan] " << callback_data->pMessage << std::endl;
  return VK_FALSE;
}

bool validation_layer_available() {
  std::uint32_t count = 0;
  vkEnumerateInstanceLayerProperties(&count, nullptr);
  std::vector<VkLayerProperties> layers(count);
  vkEnumerateInstanceLayerProperties(&count, layers.data());

  for (const auto &layer : layers)
    if (std::strcmp(layer.layerName, validation_layer_name) == 0)
      return true;
  return false;
}

} // namespace

ocean_application::ocean_application(const application_options &opts)
    : _opts{opts} {}

ocean_application::~ocean_application() { cleanup(); }

void ocean_application::run() {
  init_window();
  init_vulkan();
  init_simulation();
  main_loop();
}

// ---------------------------------------------------------------------------
// window and input
// ---------------------------------------------------------------------------

void ocean_application::init_window() {
  if (glfwInit() != GLFW_TRUE)
    throw std::runtime_error{"Could not initialize GLFW"};

  if (glfwVulkanSupported() != GLFW_TRUE)
    throw std::runtime_error{"GLFW reports that Vulkan is not supported"};

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  _window = glfwCreateWindow(static_cast<int>(_opts.window_width),
                             static_cast<int>(_opts.window_height),
                             _app_name.c_str(), nullptr, nullptr);
  if (!_window)
    throw std::runtime_error{"Could not create window"};

  glfwSetWindowUserPointer(_window, this);

  glfwSetKeyCallback(_window, [](GLFWwindow *window, int key, int, int action,
                                 int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
      return;
    auto *app =
        static_cast<ocean_application *>(glfwGetWindowUserPointer(window));
    app->on_key(key);
  });

  glfwSetMouseButtonCallback(
      _window, [](GLFWwindow *window, int button, int action, int) {
        auto *app =
            static_cast<ocean_application *>(glfwGetWindowUserPointer(window));
        app->on_mouse_button(button, action);
      });

  glfwSetCursorPosCallback(_window, [](GLFWwindow *window, double x, double y) {
    auto *app =
        static_cast<ocean_application *>(glfwGetWindowUserPointer(window));
    app->on_mouse_move(x, y);
  });

  glfwSetScrollCallback(_window, [](GLFWwindow *window, double, double y) {
    auto *app =
        static_cast<ocean_application *>(glfwGetWindowUserPointer(window));
    app->on_scroll(y);
  });
}

void ocean_application::on_key(int key) {
  switch (key) {
  case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(_window, GLFW_TRUE); break;
  case GLFW_KEY_SPACE:
    _animate = !_animate;
    std::cout << "animation is " << (_animate ? "ON" : "OFF") << std::endl;
    break;
  case GLFW_KEY_A:
    _params.wind_magnitude += 1.0f;
    _params_changed = true;
    break;
  case GLFW_KEY_Z:
    _params.wind_magnitude = std::max(1.0f, _params.wind_magnitude - 1.0f);
    _params_changed = true;
    break;
  case GLFW_KEY_S:
    _params.wind_angle += 1.0f;
    _params_changed = true;
    break;
  case GLFW_KEY_X:
    _params.wind_angle -= 1.0f;
    _params_changed = true;
    break;
  case GLFW_KEY_D:
    _params.amplitude += 0.5f;
    _params_changed = true;
    break;
  case GLFW_KEY_C:
    _params.amplitude = std::max(0.5f, _params.amplitude - 0.5f);
    _params_changed = true;
    break;
  case GLFW_KEY_F: _choppiness += 0.5f; break;
  case GLFW_KEY_V: _choppiness -= 0.5f; break;
  case GLFW_KEY_G: _alt_scale += 0.5f; break;
  case GLFW_KEY_B: _alt_scale -= 0.5f; break;
  case GLFW_KEY_W:
    if (_fill_mode_non_solid)
      _wireframe_mode = !_wireframe_mode;
    break;
  case GLFW_KEY_E: _show_fps = !_show_fps; break;
  default: break;
  }
}

void ocean_application::on_mouse_button(int button, int action) {
  if (button != GLFW_MOUSE_BUTTON_LEFT)
    return;

  if (action == GLFW_PRESS) {
    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(_window, &x, &y);
    _camera.drag = true;
    _camera.begin = glm::vec2{static_cast<float>(x), static_cast<float>(y)};
  } else if (action == GLFW_RELEASE) {
    _camera.drag = false;
  }
}

void ocean_application::on_mouse_move(double x, double y) {
  if (!_camera.drag)
    return;

  constexpr float drag_speed_fac = 0.2f;

  const glm::vec2 position{static_cast<float>(x), static_cast<float>(y)};
  const glm::vec2 offset = _camera.begin - position;
  _camera.begin = position;

  _camera.yaw -= offset.x * drag_speed_fac;
  _camera.pitch += offset.y * drag_speed_fac;
  _camera.pitch = std::clamp(_camera.pitch, -89.0f, 89.0f);

  const float yaw = glm::radians(_camera.yaw);
  const float pitch = glm::radians(_camera.pitch);

  const glm::vec3 dir{std::sin(yaw) * std::cos(pitch),
                      std::cos(yaw) * std::cos(pitch), std::sin(pitch)};

  _camera.dir = glm::normalize(dir);
  _camera.rvec =
      glm::normalize(glm::cross(_camera.dir, glm::vec3{0.0f, 0.0f, 1.0f}));
  _camera.up = glm::normalize(glm::cross(_camera.rvec, _camera.dir));
}

void ocean_application::on_scroll(double offset) {
  constexpr float roll_speed_fac = 10.0f;
  _camera.eye += _camera.dir * static_cast<float>(offset) * roll_speed_fac;
}

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------

void ocean_application::init_vulkan() {
  create_instance();
  create_surface();
  pick_physical_device();
  create_logical_device();
  create_swap_chain();
  create_swap_chain_image_views();
  create_render_pass();
  create_descriptor_set_layout();
  create_graphics_pipelines();
  create_command_pool();
  create_depth_resources();
  create_framebuffers();
  create_grid_buffers();
  create_textures();
  create_texture_sampler();
  create_uniform_buffers();
  create_descriptor_pool();
  create_descriptor_sets();
  create_command_buffers();
  create_sync_objects();
}

void ocean_application::init_simulation() {
  _simulation = std::make_unique<ocean_simulation>(
      _opts.texture_size,
      static_cast<float>(_opts.grid_size) * _opts.mesh_spacing);

  std::cout << "Running the ocean simulation on SYCL device: "
            << _simulation->device_name() << std::endl;

  const VkDeviceSize staging_size =
      static_cast<VkDeviceSize>(_simulation->map_size_in_bytes()) *
      texture_count;

  create_buffer(staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                _staging_buffer, _staging_buffer_memory);
  check(vkMapMemory(_device, _staging_buffer_memory, 0, staging_size, 0,
                    &_staging_buffer_mapped),
        "Could not map staging buffer");
}

void ocean_application::create_instance() {
  if (_opts.validation_layers && !validation_layer_available()) {
    std::cerr << "Validation layers requested but not available, disabling."
              << std::endl;
    _opts.validation_layers = false;
  }

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = _app_name.c_str();
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "AdaptiveCpp samples";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_0;

  std::uint32_t glfw_extension_count = 0;
  const char **glfw_extensions =
      glfwGetRequiredInstanceExtensions(&glfw_extension_count);
  if (!glfw_extensions)
    throw std::runtime_error{
        "GLFW could not determine the required Vulkan instance extensions"};

  std::vector<const char *> extensions{glfw_extensions,
                                       glfw_extensions + glfw_extension_count};
  if (_opts.validation_layers)
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount =
      static_cast<std::uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

  VkDebugUtilsMessengerCreateInfoEXT debug_info{};
  if (_opts.validation_layers) {
    create_info.enabledLayerCount = 1;
    create_info.ppEnabledLayerNames = &validation_layer_name;

    debug_info.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_info.pfnUserCallback = debug_callback;
    create_info.pNext = &debug_info;
  }

  check(vkCreateInstance(&create_info, nullptr, &_instance),
        "Could not create Vulkan instance");

  if (_opts.validation_layers) {
    auto create_messenger =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(_instance,
                                  "vkCreateDebugUtilsMessengerEXT"));
    if (create_messenger)
      check(create_messenger(_instance, &debug_info, nullptr,
                             &_debug_messenger),
            "Could not create debug messenger");
  }
}

void ocean_application::create_surface() {
  check(glfwCreateWindowSurface(_instance, _window, nullptr, &_surface),
        "Could not create window surface");
}

void ocean_application::pick_physical_device() {
  std::uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(_instance, &device_count, nullptr);
  if (device_count == 0)
    throw std::runtime_error{"No Vulkan capable device found"};

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(_instance, &device_count, devices.data());

  for (VkPhysicalDevice device : devices) {
    std::uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             families.data());

    bool has_graphics = false;
    bool has_present = false;
    std::uint32_t graphics_family = 0;
    std::uint32_t present_family = 0;

    for (std::uint32_t i = 0; i < queue_family_count; ++i) {
      if (!has_graphics && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
        has_graphics = true;
        graphics_family = i;
      }

      VkBool32 present_support = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface,
                                           &present_support);
      if (!has_present && present_support == VK_TRUE) {
        has_present = true;
        present_family = i;
      }
    }

    if (!has_graphics || !has_present)
      continue;

    std::uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                         nullptr);
    std::vector<VkExtensionProperties> available(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                         available.data());

    bool has_swap_chain = false;
    for (const auto &extension : available)
      if (std::strcmp(extension.extensionName,
                      VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
        has_swap_chain = true;

    if (!has_swap_chain)
      continue;

    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(device, &features);

    VkFormatProperties format_properties{};
    vkGetPhysicalDeviceFormatProperties(device, interop_texture_format,
                                        &format_properties);
    if (!(format_properties.optimalTilingFeatures &
          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
      continue;

    _physical_device = device;
    _graphics_family = graphics_family;
    _present_family = present_family;
    _fill_mode_non_solid = features.fillModeNonSolid == VK_TRUE;
    break;
  }

  if (_physical_device == VK_NULL_HANDLE)
    throw std::runtime_error{"No suitable Vulkan device found"};

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(_physical_device, &properties);
  std::cout << "Rendering on Vulkan device: " << properties.deviceName
            << std::endl;
}

void ocean_application::create_logical_device() {
  const std::set<std::uint32_t> unique_families{_graphics_family,
                                                _present_family};
  const float queue_priority = 1.0f;

  std::vector<VkDeviceQueueCreateInfo> queue_infos;
  for (std::uint32_t family : unique_families) {
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;
    queue_infos.push_back(queue_info);
  }

  VkPhysicalDeviceFeatures features{};
  features.fillModeNonSolid = _fill_mode_non_solid ? VK_TRUE : VK_FALSE;

  const char *const device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.queueCreateInfoCount =
      static_cast<std::uint32_t>(queue_infos.size());
  create_info.pQueueCreateInfos = queue_infos.data();
  create_info.pEnabledFeatures = &features;
  create_info.enabledExtensionCount = 1;
  create_info.ppEnabledExtensionNames = device_extensions;
  if (_opts.validation_layers) {
    create_info.enabledLayerCount = 1;
    create_info.ppEnabledLayerNames = &validation_layer_name;
  }

  check(vkCreateDevice(_physical_device, &create_info, nullptr, &_device),
        "Could not create logical device");

  vkGetDeviceQueue(_device, _graphics_family, 0, &_graphics_queue);
  vkGetDeviceQueue(_device, _present_family, 0, &_present_queue);
}

void ocean_application::create_swap_chain() {
  VkSurfaceCapabilitiesKHR capabilities{};
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_physical_device, _surface,
                                            &capabilities);

  std::uint32_t format_count = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(_physical_device, _surface,
                                       &format_count, nullptr);
  if (format_count == 0)
    throw std::runtime_error{"Surface does not support any format"};
  std::vector<VkSurfaceFormatKHR> formats(format_count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(_physical_device, _surface,
                                       &format_count, formats.data());

  VkSurfaceFormatKHR surface_format = formats[0];
  for (const auto &format : formats)
    if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
      surface_format = format;

  std::uint32_t present_mode_count = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(_physical_device, _surface,
                                            &present_mode_count, nullptr);
  std::vector<VkPresentModeKHR> present_modes(present_mode_count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      _physical_device, _surface, &present_mode_count, present_modes.data());

  VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
  if (_opts.immediate)
    for (VkPresentModeKHR mode : present_modes)
      if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
        present_mode = mode;

  VkExtent2D extent = capabilities.currentExtent;
  if (capabilities.currentExtent.width == UINT32_MAX) {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(_window, &width, &height);
    extent.width = std::clamp(static_cast<std::uint32_t>(width),
                              capabilities.minImageExtent.width,
                              capabilities.maxImageExtent.width);
    extent.height = std::clamp(static_cast<std::uint32_t>(height),
                               capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height);
  }

  std::uint32_t image_count = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount)
    image_count = capabilities.maxImageCount;

  VkSwapchainCreateInfoKHR create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface = _surface;
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  const std::uint32_t families[] = {_graphics_family, _present_family};
  if (_graphics_family != _present_family) {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = families;
  } else {
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  create_info.preTransform = capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;
  create_info.oldSwapchain = VK_NULL_HANDLE;

  check(vkCreateSwapchainKHR(_device, &create_info, nullptr, &_swap_chain),
        "Could not create swap chain");

  std::uint32_t actual_image_count = 0;
  vkGetSwapchainImagesKHR(_device, _swap_chain, &actual_image_count, nullptr);
  _swap_chain_images.resize(actual_image_count);
  vkGetSwapchainImagesKHR(_device, _swap_chain, &actual_image_count,
                          _swap_chain_images.data());

  _swap_chain_image_format = surface_format.format;
  _swap_chain_extent = extent;
}

void ocean_application::create_swap_chain_image_views() {
  _swap_chain_image_views.resize(_swap_chain_images.size());
  for (std::size_t i = 0; i < _swap_chain_images.size(); ++i)
    _swap_chain_image_views[i] =
        create_image_view(_swap_chain_images[i], _swap_chain_image_format,
                          VK_IMAGE_ASPECT_COLOR_BIT);
}

void ocean_application::create_render_pass() {
  VkAttachmentDescription color_attachment{};
  color_attachment.format = _swap_chain_image_format;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentDescription depth_attachment{};
  depth_attachment.format = find_depth_format();
  depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depth_attachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference color_reference{};
  color_reference.attachment = 0;
  color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depth_reference{};
  depth_reference.attachment = 1;
  depth_reference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_reference;
  subpass.pDepthStencilAttachment = &depth_reference;

  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  const std::array<VkAttachmentDescription, 2> attachments{color_attachment,
                                                           depth_attachment};

  VkRenderPassCreateInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount =
      static_cast<std::uint32_t>(attachments.size());
  render_pass_info.pAttachments = attachments.data();
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies = &dependency;

  check(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_render_pass),
        "Could not create render pass");
}

void ocean_application::create_descriptor_set_layout() {
  VkDescriptorSetLayoutBinding displacement_binding{};
  displacement_binding.binding = 0;
  displacement_binding.descriptorCount = 1;
  displacement_binding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  displacement_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutBinding normal_binding{};
  normal_binding.binding = 1;
  normal_binding.descriptorCount = 1;
  normal_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  normal_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutBinding uniform_binding{};
  uniform_binding.binding = 2;
  uniform_binding.descriptorCount = 1;
  uniform_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uniform_binding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  const std::array<VkDescriptorSetLayoutBinding, 3> bindings{
      displacement_binding, normal_binding, uniform_binding};

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();

  check(vkCreateDescriptorSetLayout(_device, &layout_info, nullptr,
                                    &_descriptor_set_layout),
        "Could not create descriptor set layout");
}

void ocean_application::create_graphics_pipelines() {
  const std::vector<char> vertex_code = read_shader("ocean.vert.spv");
  const std::vector<char> fragment_code = read_shader("ocean.frag.spv");

  VkShaderModule vertex_module = create_shader_module(vertex_code);
  VkShaderModule fragment_module = create_shader_module(fragment_code);

  VkPipelineShaderStageCreateInfo vertex_stage{};
  vertex_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertex_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertex_stage.module = vertex_module;
  vertex_stage.pName = "main";

  VkPipelineShaderStageCreateInfo fragment_stage{};
  fragment_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragment_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragment_stage.module = fragment_module;
  fragment_stage.pName = "main";

  const std::array<VkPipelineShaderStageCreateInfo, 2> stages{vertex_stage,
                                                              fragment_stage};

  VkVertexInputBindingDescription binding_description{};
  binding_description.binding = 0;
  binding_description.stride = sizeof(vertex);
  binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions{};
  attribute_descriptions[0].binding = 0;
  attribute_descriptions[0].location = 0;
  attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attribute_descriptions[0].offset = offsetof(vertex, position);
  attribute_descriptions[1].binding = 0;
  attribute_descriptions[1].location = 1;
  attribute_descriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
  attribute_descriptions[1].offset = offsetof(vertex, tex_coord);

  VkPipelineVertexInputStateCreateInfo vertex_input{};
  vertex_input.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input.vertexBindingDescriptionCount = 1;
  vertex_input.pVertexBindingDescriptions = &binding_description;
  vertex_input.vertexAttributeDescriptionCount =
      static_cast<std::uint32_t>(attribute_descriptions.size());
  vertex_input.pVertexAttributeDescriptions = attribute_descriptions.data();

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  input_assembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(_swap_chain_extent.width);
  viewport.height = static_cast<float>(_swap_chain_extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = _swap_chain_extent;

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depth_stencil{};
  depth_stencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable = VK_TRUE;
  depth_stencil.depthWriteEnable = VK_TRUE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState color_blend_attachment{};
  color_blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo color_blending{};
  color_blending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &color_blend_attachment;

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &_descriptor_set_layout;

  check(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr,
                               &_pipeline_layout),
        "Could not create pipeline layout");

  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = static_cast<std::uint32_t>(stages.size());
  pipeline_info.pStages = stages.data();
  pipeline_info.pVertexInputState = &vertex_input;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = &depth_stencil;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.layout = _pipeline_layout;
  pipeline_info.renderPass = _render_pass;
  pipeline_info.subpass = 0;

  check(vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info,
                                  nullptr, &_graphics_pipeline),
        "Could not create graphics pipeline");

  if (_fill_mode_non_solid) {
    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    check(vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info,
                                    nullptr, &_wireframe_pipeline),
          "Could not create wireframe pipeline");
  }

  vkDestroyShaderModule(_device, vertex_module, nullptr);
  vkDestroyShaderModule(_device, fragment_module, nullptr);
}

void ocean_application::create_depth_resources() {
  const VkFormat depth_format = find_depth_format();

  create_image(_swap_chain_extent.width, _swap_chain_extent.height,
               depth_format, VK_IMAGE_TILING_OPTIMAL,
               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _depth_image,
               _depth_image_memory);
  _depth_image_view = create_image_view(_depth_image, depth_format,
                                        VK_IMAGE_ASPECT_DEPTH_BIT);
}

void ocean_application::create_framebuffers() {
  _swap_chain_framebuffers.resize(_swap_chain_image_views.size());

  for (std::size_t i = 0; i < _swap_chain_image_views.size(); ++i) {
    const std::array<VkImageView, 2> attachments{_swap_chain_image_views[i],
                                                 _depth_image_view};

    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = _render_pass;
    framebuffer_info.attachmentCount =
        static_cast<std::uint32_t>(attachments.size());
    framebuffer_info.pAttachments = attachments.data();
    framebuffer_info.width = _swap_chain_extent.width;
    framebuffer_info.height = _swap_chain_extent.height;
    framebuffer_info.layers = 1;

    check(vkCreateFramebuffer(_device, &framebuffer_info, nullptr,
                              &_swap_chain_framebuffers[i]),
          "Could not create framebuffer");
  }
}

void ocean_application::create_command_pool() {
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = _graphics_family;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  check(vkCreateCommandPool(_device, &pool_info, nullptr, &_command_pool),
        "Could not create command pool");
}

void ocean_application::create_grid_buffers() {
  const std::uint32_t n = _opts.grid_size;
  const float offset = 0.5f * static_cast<float>(n - 1) * _opts.mesh_spacing;

  _grid_vertices.reserve(static_cast<std::size_t>(n) * n);
  for (std::uint32_t y = 0; y < n; ++y) {
    for (std::uint32_t x = 0; x < n; ++x) {
      vertex v{};
      v.position = glm::vec3{static_cast<float>(x) * _opts.mesh_spacing - offset,
                             static_cast<float>(y) * _opts.mesh_spacing - offset,
                             0.0f};
      v.tex_coord = glm::vec2{static_cast<float>(x) / static_cast<float>(n - 1),
                              static_cast<float>(y) / static_cast<float>(n - 1)};
      _grid_vertices.push_back(v);
    }
  }

  _grid_indices.reserve(static_cast<std::size_t>(n - 1) * (n - 1) * 6);
  for (std::uint32_t y = 0; y + 1 < n; ++y) {
    for (std::uint32_t x = 0; x + 1 < n; ++x) {
      const std::uint32_t i0 = y * n + x;
      const std::uint32_t i1 = i0 + 1;
      const std::uint32_t i2 = i0 + n;
      const std::uint32_t i3 = i2 + 1;

      _grid_indices.push_back(i0);
      _grid_indices.push_back(i2);
      _grid_indices.push_back(i1);

      _grid_indices.push_back(i1);
      _grid_indices.push_back(i2);
      _grid_indices.push_back(i3);
    }
  }

  auto upload = [this](const void *data, VkDeviceSize size,
                       VkBufferUsageFlags usage, VkBuffer &buffer,
                       VkDeviceMemory &memory) {
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  staging_buffer, staging_memory);

    void *mapped = nullptr;
    check(vkMapMemory(_device, staging_memory, 0, size, 0, &mapped),
          "Could not map staging memory");
    std::memcpy(mapped, data, static_cast<std::size_t>(size));
    vkUnmapMemory(_device, staging_memory);

    create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);
    copy_buffer(staging_buffer, buffer, size);

    vkDestroyBuffer(_device, staging_buffer, nullptr);
    vkFreeMemory(_device, staging_memory, nullptr);
  };

  upload(_grid_vertices.data(),
         static_cast<VkDeviceSize>(_grid_vertices.size() * sizeof(vertex)),
         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, _vertex_buffer,
         _vertex_buffer_memory);
  upload(_grid_indices.data(),
         static_cast<VkDeviceSize>(_grid_indices.size() *
                                   sizeof(std::uint32_t)),
         VK_BUFFER_USAGE_INDEX_BUFFER_BIT, _index_buffer,
         _index_buffer_memory);
}

void ocean_application::create_textures() {
  const std::size_t image_count = _swap_chain_images.size();

  for (std::size_t target = 0; target < texture_count; ++target) {
    _textures[target].images.resize(image_count, VK_NULL_HANDLE);
    _textures[target].memories.resize(image_count, VK_NULL_HANDLE);
    _textures[target].views.resize(image_count, VK_NULL_HANDLE);

    for (std::size_t i = 0; i < image_count; ++i) {
      create_image(_opts.texture_size, _opts.texture_size,
                   interop_texture_format, VK_IMAGE_TILING_OPTIMAL,
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   _textures[target].images[i], _textures[target].memories[i]);

      transition_image_layout(_textures[target].images[i],
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

      _textures[target].views[i] =
          create_image_view(_textures[target].images[i],
                            interop_texture_format, VK_IMAGE_ASPECT_COLOR_BIT);
    }
  }
}

void ocean_application::create_texture_sampler() {
  for (std::size_t target = 0; target < texture_count; ++target) {
    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.maxAnisotropy = 1.0f;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    check(vkCreateSampler(_device, &sampler_info, nullptr,
                          &_texture_samplers[target]),
          "Could not create texture sampler");
  }
}

void ocean_application::create_uniform_buffers() {
  const VkDeviceSize size = sizeof(uniform_buffer_object);
  const std::size_t image_count = _swap_chain_images.size();

  _uniform_buffers.resize(image_count, VK_NULL_HANDLE);
  _uniform_buffer_memories.resize(image_count, VK_NULL_HANDLE);
  _uniform_buffers_mapped.resize(image_count, nullptr);

  for (std::size_t i = 0; i < image_count; ++i) {
    create_buffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  _uniform_buffers[i], _uniform_buffer_memories[i]);
    check(vkMapMemory(_device, _uniform_buffer_memories[i], 0, size, 0,
                      &_uniform_buffers_mapped[i]),
          "Could not map uniform buffer");
  }
}

void ocean_application::create_descriptor_pool() {
  const std::uint32_t image_count =
      static_cast<std::uint32_t>(_swap_chain_images.size());

  std::array<VkDescriptorPoolSize, 2> pool_sizes{};
  pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  pool_sizes[0].descriptorCount = image_count * texture_count;
  pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  pool_sizes[1].descriptorCount = image_count;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
  pool_info.pPoolSizes = pool_sizes.data();
  pool_info.maxSets = image_count;

  check(vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptor_pool),
        "Could not create descriptor pool");
}

void ocean_application::create_descriptor_sets() {
  const std::size_t image_count = _swap_chain_images.size();
  const std::vector<VkDescriptorSetLayout> layouts(image_count,
                                                   _descriptor_set_layout);

  VkDescriptorSetAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocate_info.descriptorPool = _descriptor_pool;
  allocate_info.descriptorSetCount = static_cast<std::uint32_t>(image_count);
  allocate_info.pSetLayouts = layouts.data();

  _descriptor_sets.resize(image_count);
  check(vkAllocateDescriptorSets(_device, &allocate_info,
                                 _descriptor_sets.data()),
        "Could not allocate descriptor sets");

  for (std::size_t i = 0; i < image_count; ++i) {
    VkDescriptorImageInfo displacement_info{};
    displacement_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    displacement_info.imageView = _textures[texture_displacement].views[i];
    displacement_info.sampler = _texture_samplers[texture_displacement];

    VkDescriptorImageInfo normal_info{};
    normal_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    normal_info.imageView = _textures[texture_normal_map].views[i];
    normal_info.sampler = _texture_samplers[texture_normal_map];

    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = _uniform_buffers[i];
    buffer_info.offset = 0;
    buffer_info.range = sizeof(uniform_buffer_object);

    std::array<VkWriteDescriptorSet, 3> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = _descriptor_sets[i];
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &displacement_info;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = _descriptor_sets[i];
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &normal_info;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = _descriptor_sets[i];
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(_device, static_cast<std::uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
  }
}

void ocean_application::create_command_buffers() {
  _command_buffers.resize(_swap_chain_images.size());

  VkCommandBufferAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate_info.commandPool = _command_pool;
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate_info.commandBufferCount =
      static_cast<std::uint32_t>(_command_buffers.size());

  check(vkAllocateCommandBuffers(_device, &allocate_info,
                                 _command_buffers.data()),
        "Could not allocate command buffers");
}

void ocean_application::create_sync_objects() {
  _image_available_semaphores.resize(max_frames_in_flight);
  _render_finished_semaphores.resize(_swap_chain_images.size());
  _in_flight_fences.resize(max_frames_in_flight);
  _images_in_flight.resize(_swap_chain_images.size(), VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (std::size_t i = 0; i < max_frames_in_flight; ++i) {
    check(vkCreateSemaphore(_device, &semaphore_info, nullptr,
                            &_image_available_semaphores[i]),
          "Could not create semaphore");
    check(vkCreateFence(_device, &fence_info, nullptr, &_in_flight_fences[i]),
          "Could not create fence");
  }

  for (std::size_t i = 0; i < _swap_chain_images.size(); ++i)
    check(vkCreateSemaphore(_device, &semaphore_info, nullptr,
                            &_render_finished_semaphores[i]),
          "Could not create semaphore");
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

void ocean_application::main_loop() {
  while (glfwWindowShouldClose(_window) == GLFW_FALSE) {
    glfwPollEvents();
    draw_frame();
  }

  vkDeviceWaitIdle(_device);
}

void ocean_application::draw_frame() {
  vkWaitForFences(_device, 1, &_in_flight_fences[_current_frame], VK_TRUE,
                  UINT64_MAX);

  std::uint32_t image_index = 0;
  const VkResult acquire_result = vkAcquireNextImageKHR(
      _device, _swap_chain, UINT64_MAX,
      _image_available_semaphores[_current_frame], VK_NULL_HANDLE,
      &image_index);
  if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR)
    check(acquire_result, "Could not acquire swap chain image");

  if (_images_in_flight[image_index] != VK_NULL_HANDLE)
    vkWaitForFences(_device, 1, &_images_in_flight[image_index], VK_TRUE,
                    UINT64_MAX);
  _images_in_flight[image_index] = _in_flight_fences[_current_frame];

  update_simulation(image_index);
  update_uniforms(image_index);
  record_command_buffer(image_index);

  const VkPipelineStageFlags wait_stage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = &_image_available_semaphores[_current_frame];
  submit_info.pWaitDstStageMask = &wait_stage;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &_command_buffers[image_index];
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &_render_finished_semaphores[image_index];

  vkResetFences(_device, 1, &_in_flight_fences[_current_frame]);
  check(vkQueueSubmit(_graphics_queue, 1, &submit_info,
                      _in_flight_fences[_current_frame]),
        "Could not submit draw command buffer");

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &_render_finished_semaphores[image_index];
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &_swap_chain;
  present_info.pImageIndices = &image_index;

  const VkResult present_result =
      vkQueuePresentKHR(_present_queue, &present_info);
  if (present_result != VK_SUCCESS && present_result != VK_SUBOPTIMAL_KHR)
    check(present_result, "Could not present swap chain image");

  _current_frame = (_current_frame + 1) % max_frames_in_flight;

  update_window_title();
}

void ocean_application::update_simulation(std::uint32_t image_index) {
  if (_params_changed) {
    _simulation->set_parameters(_params);
    _params_changed = false;
  }

  const auto now = std::chrono::steady_clock::now();
  if (_animate)
    _elapsed += std::chrono::duration<float>(now - _last_update).count();
  _last_update = now;

  _simulation->update(_elapsed);

  const VkDeviceSize map_size =
      static_cast<VkDeviceSize>(_simulation->map_size_in_bytes());
  auto *staging = static_cast<unsigned char *>(_staging_buffer_mapped);

  _simulation->copy_displacement_map(staging);
  _simulation->copy_normal_map(staging + map_size);

  for (std::size_t target = 0; target < texture_count; ++target) {
    VkImage image = _textures[target].images[image_index];

    transition_image_layout(image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    copy_buffer_to_image(_staging_buffer, map_size * target, image,
                         _opts.texture_size, _opts.texture_size);

    transition_image_layout(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
}

void ocean_application::update_uniforms(std::uint32_t image_index) {
  constexpr float sun_elevation = 25.0f;
  constexpr float sun_azimuth = 90.0f;

  uniform_buffer_object ubo{};
  ubo.view = glm::lookAt(_camera.eye, _camera.eye + _camera.dir, _camera.up);
  ubo.projection = glm::perspective(
      glm::radians(45.0f),
      static_cast<float>(_swap_chain_extent.width) /
          static_cast<float>(_swap_chain_extent.height),
      1.0f, 10000.0f);
  // GLM was designed for OpenGL, where the Y coordinate of the clip space is
  // inverted compared to Vulkan.
  ubo.projection[1][1] *= -1.0f;

  const float elevation = glm::radians(sun_elevation);
  const float azimuth = glm::radians(sun_azimuth);
  ubo.sun_dir = glm::normalize(glm::vec3{std::cos(elevation) * std::cos(azimuth),
                                         std::cos(elevation) * std::sin(azimuth),
                                         std::sin(elevation)});
  ubo.choppiness = _choppiness;
  ubo.alt_scale = _alt_scale;

  std::memcpy(_uniform_buffers_mapped[image_index], &ubo, sizeof(ubo));
}

void ocean_application::record_command_buffer(std::uint32_t image_index) {
  VkCommandBuffer command_buffer = _command_buffers[image_index];
  check(vkResetCommandBuffer(command_buffer, 0),
        "Could not reset command buffer");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  check(vkBeginCommandBuffer(command_buffer, &begin_info),
        "Could not begin command buffer");

  std::array<VkClearValue, 2> clear_values{};
  clear_values[0].color = {{0.0f, 0.1f, 0.2f, 1.0f}};
  clear_values[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_pass_info.renderPass = _render_pass;
  render_pass_info.framebuffer = _swap_chain_framebuffers[image_index];
  render_pass_info.renderArea.offset = {0, 0};
  render_pass_info.renderArea.extent = _swap_chain_extent;
  render_pass_info.clearValueCount =
      static_cast<std::uint32_t>(clear_values.size());
  render_pass_info.pClearValues = clear_values.data();

  vkCmdBeginRenderPass(command_buffer, &render_pass_info,
                       VK_SUBPASS_CONTENTS_INLINE);

  const bool wireframe =
      _wireframe_mode && _wireframe_pipeline != VK_NULL_HANDLE;
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    wireframe ? _wireframe_pipeline : _graphics_pipeline);

  const VkDeviceSize vertex_offset = 0;
  vkCmdBindVertexBuffers(command_buffer, 0, 1, &_vertex_buffer,
                         &vertex_offset);
  vkCmdBindIndexBuffer(command_buffer, _index_buffer, 0, VK_INDEX_TYPE_UINT32);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          _pipeline_layout, 0, 1,
                          &_descriptor_sets[image_index], 0, nullptr);
  vkCmdDrawIndexed(command_buffer,
                   static_cast<std::uint32_t>(_grid_indices.size()), 1, 0, 0,
                   0);

  vkCmdEndRenderPass(command_buffer);
  check(vkEndCommandBuffer(command_buffer), "Could not record command buffer");
}

void ocean_application::update_window_title() {
  ++_delta_frames;

  const auto now = std::chrono::steady_clock::now();
  const float seconds =
      std::chrono::duration<float>(now - _fps_last_time).count();
  if (seconds < 1.0f)
    return;

  std::string title = _app_name;
  if (_show_fps) {
    const float fps = static_cast<float>(_delta_frames) / seconds;
    title += " [" + std::to_string(static_cast<int>(fps)) + " fps, wind " +
             std::to_string(static_cast<int>(_params.wind_magnitude)) +
             " m/s]";
  }
  glfwSetWindowTitle(_window, title.c_str());

  _fps_last_time = now;
  _delta_frames = 0;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

std::uint32_t
ocean_application::find_memory_type(std::uint32_t type_filter,
                                    VkMemoryPropertyFlags properties) const {
  VkPhysicalDeviceMemoryProperties memory_properties{};
  vkGetPhysicalDeviceMemoryProperties(_physical_device, &memory_properties);

  for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
    if ((type_filter & (1u << i)) &&
        (memory_properties.memoryTypes[i].propertyFlags & properties) ==
            properties)
      return i;

  throw std::runtime_error{"Could not find suitable memory type"};
}

void ocean_application::create_buffer(VkDeviceSize size,
                                      VkBufferUsageFlags usage,
                                      VkMemoryPropertyFlags properties,
                                      VkBuffer &buffer,
                                      VkDeviceMemory &memory) {
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  check(vkCreateBuffer(_device, &buffer_info, nullptr, &buffer),
        "Could not create buffer");

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(_device, buffer, &requirements);

  VkMemoryAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex =
      find_memory_type(requirements.memoryTypeBits, properties);

  check(vkAllocateMemory(_device, &allocate_info, nullptr, &memory),
        "Could not allocate buffer memory");
  check(vkBindBufferMemory(_device, buffer, memory, 0),
        "Could not bind buffer memory");
}

void ocean_application::copy_buffer(VkBuffer src, VkBuffer dst,
                                    VkDeviceSize size) {
  VkCommandBuffer command_buffer = begin_single_time_commands();

  VkBufferCopy region{};
  region.size = size;
  vkCmdCopyBuffer(command_buffer, src, dst, 1, &region);

  end_single_time_commands(command_buffer);
}

void ocean_application::create_image(std::uint32_t width, std::uint32_t height,
                                     VkFormat format, VkImageTiling tiling,
                                     VkImageUsageFlags usage,
                                     VkMemoryPropertyFlags properties,
                                     VkImage &image, VkDeviceMemory &memory) {
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.extent = {width, height, 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.format = format;
  image_info.tiling = tiling;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage = usage;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  check(vkCreateImage(_device, &image_info, nullptr, &image),
        "Could not create image");

  VkMemoryRequirements requirements{};
  vkGetImageMemoryRequirements(_device, image, &requirements);

  VkMemoryAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex =
      find_memory_type(requirements.memoryTypeBits, properties);

  check(vkAllocateMemory(_device, &allocate_info, nullptr, &memory),
        "Could not allocate image memory");
  check(vkBindImageMemory(_device, image, memory, 0),
        "Could not bind image memory");
}

VkImageView ocean_application::create_image_view(
    VkImage image, VkFormat format, VkImageAspectFlags aspect_flags) {
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  view_info.subresourceRange.aspectMask = aspect_flags;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;

  VkImageView view = VK_NULL_HANDLE;
  check(vkCreateImageView(_device, &view_info, nullptr, &view),
        "Could not create image view");
  return view;
}

void ocean_application::transition_image_layout(VkImage image,
                                                VkImageLayout old_layout,
                                                VkImageLayout new_layout) {
  VkCommandBuffer command_buffer = begin_single_time_commands();

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  VkPipelineStageFlags destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

  if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    source_stage = (old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        ? VK_ACCESS_TRANSFER_WRITE_BIT
        : 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    source_stage = (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        ? VK_PIPELINE_STAGE_TRANSFER_BIT
        : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destination_stage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
  } else {
    throw std::invalid_argument{"Unsupported image layout transition"};
  }

  vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);

  end_single_time_commands(command_buffer);
}

void ocean_application::copy_buffer_to_image(VkBuffer buffer,
                                             VkDeviceSize buffer_offset,
                                             VkImage image,
                                             std::uint32_t width,
                                             std::uint32_t height) {
  VkCommandBuffer command_buffer = begin_single_time_commands();

  VkBufferImageCopy region{};
  region.bufferOffset = buffer_offset;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {width, height, 1};

  vkCmdCopyBufferToImage(command_buffer, buffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  end_single_time_commands(command_buffer);
}

VkCommandBuffer ocean_application::begin_single_time_commands() {
  VkCommandBufferAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate_info.commandPool = _command_pool;
  allocate_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  check(vkAllocateCommandBuffers(_device, &allocate_info, &command_buffer),
        "Could not allocate command buffer");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(command_buffer, &begin_info),
        "Could not begin command buffer");

  return command_buffer;
}

void ocean_application::end_single_time_commands(
    VkCommandBuffer command_buffer) {
  check(vkEndCommandBuffer(command_buffer), "Could not end command buffer");

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  check(vkQueueSubmit(_graphics_queue, 1, &submit_info, VK_NULL_HANDLE),
        "Could not submit command buffer");
  check(vkQueueWaitIdle(_graphics_queue), "Could not wait for queue");

  vkFreeCommandBuffers(_device, _command_pool, 1, &command_buffer);
}

VkShaderModule
ocean_application::create_shader_module(const std::vector<char> &code) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size();
  create_info.pCode = reinterpret_cast<const std::uint32_t *>(code.data());

  VkShaderModule module = VK_NULL_HANDLE;
  check(vkCreateShaderModule(_device, &create_info, nullptr, &module),
        "Could not create shader module");
  return module;
}

VkFormat ocean_application::find_depth_format() const {
  const VkFormat candidates[] = {VK_FORMAT_D32_SFLOAT,
                                 VK_FORMAT_D32_SFLOAT_S8_UINT,
                                 VK_FORMAT_D24_UNORM_S8_UINT};

  for (VkFormat format : candidates) {
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(_physical_device, format, &properties);
    if (properties.optimalTilingFeatures &
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
      return format;
  }

  throw std::runtime_error{"Could not find a supported depth format"};
}

void ocean_application::cleanup() {
  if (_device != VK_NULL_HANDLE)
    vkDeviceWaitIdle(_device);

  // the SYCL simulation must be destroyed before the Vulkan resources it
  // writes into
  _simulation.reset();

  if (_device != VK_NULL_HANDLE) {
    for (VkSemaphore semaphore : _render_finished_semaphores)
      vkDestroySemaphore(_device, semaphore, nullptr);
    for (VkSemaphore semaphore : _image_available_semaphores)
      vkDestroySemaphore(_device, semaphore, nullptr);
    for (VkFence fence : _in_flight_fences)
      vkDestroyFence(_device, fence, nullptr);

    if (_descriptor_pool != VK_NULL_HANDLE)
      vkDestroyDescriptorPool(_device, _descriptor_pool, nullptr);

    for (std::size_t i = 0; i < _uniform_buffers.size(); ++i) {
      if (_uniform_buffers_mapped[i])
        vkUnmapMemory(_device, _uniform_buffer_memories[i]);
      vkDestroyBuffer(_device, _uniform_buffers[i], nullptr);
      vkFreeMemory(_device, _uniform_buffer_memories[i], nullptr);
    }

    for (std::size_t target = 0; target < texture_count; ++target) {
      if (_texture_samplers[target] != VK_NULL_HANDLE)
        vkDestroySampler(_device, _texture_samplers[target], nullptr);
      for (std::size_t i = 0; i < _textures[target].images.size(); ++i) {
        vkDestroyImageView(_device, _textures[target].views[i], nullptr);
        vkDestroyImage(_device, _textures[target].images[i], nullptr);
        vkFreeMemory(_device, _textures[target].memories[i], nullptr);
      }
    }

    if (_staging_buffer_mapped)
      vkUnmapMemory(_device, _staging_buffer_memory);
    if (_staging_buffer != VK_NULL_HANDLE)
      vkDestroyBuffer(_device, _staging_buffer, nullptr);
    if (_staging_buffer_memory != VK_NULL_HANDLE)
      vkFreeMemory(_device, _staging_buffer_memory, nullptr);

    if (_index_buffer != VK_NULL_HANDLE)
      vkDestroyBuffer(_device, _index_buffer, nullptr);
    if (_index_buffer_memory != VK_NULL_HANDLE)
      vkFreeMemory(_device, _index_buffer_memory, nullptr);
    if (_vertex_buffer != VK_NULL_HANDLE)
      vkDestroyBuffer(_device, _vertex_buffer, nullptr);
    if (_vertex_buffer_memory != VK_NULL_HANDLE)
      vkFreeMemory(_device, _vertex_buffer_memory, nullptr);

    for (VkFramebuffer framebuffer : _swap_chain_framebuffers)
      vkDestroyFramebuffer(_device, framebuffer, nullptr);

    if (_depth_image_view != VK_NULL_HANDLE)
      vkDestroyImageView(_device, _depth_image_view, nullptr);
    if (_depth_image != VK_NULL_HANDLE)
      vkDestroyImage(_device, _depth_image, nullptr);
    if (_depth_image_memory != VK_NULL_HANDLE)
      vkFreeMemory(_device, _depth_image_memory, nullptr);

    if (_wireframe_pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(_device, _wireframe_pipeline, nullptr);
    if (_graphics_pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(_device, _graphics_pipeline, nullptr);
    if (_pipeline_layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
    if (_descriptor_set_layout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(_device, _descriptor_set_layout, nullptr);
    if (_render_pass != VK_NULL_HANDLE)
      vkDestroyRenderPass(_device, _render_pass, nullptr);

    for (VkImageView view : _swap_chain_image_views)
      vkDestroyImageView(_device, view, nullptr);
    if (_swap_chain != VK_NULL_HANDLE)
      vkDestroySwapchainKHR(_device, _swap_chain, nullptr);

    if (_command_pool != VK_NULL_HANDLE)
      vkDestroyCommandPool(_device, _command_pool, nullptr);

    vkDestroyDevice(_device, nullptr);
    _device = VK_NULL_HANDLE;
  }

  if (_surface != VK_NULL_HANDLE)
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
  if (_debug_messenger != VK_NULL_HANDLE) {
    auto destroy_messenger =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(_instance,
                                  "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy_messenger)
      destroy_messenger(_instance, _debug_messenger, nullptr);
    _debug_messenger = VK_NULL_HANDLE;
  }
  if (_instance != VK_NULL_HANDLE)
    vkDestroyInstance(_instance, nullptr);
  _instance = VK_NULL_HANDLE;

  if (_window) {
    glfwDestroyWindow(_window);
    _window = nullptr;
  }
  glfwTerminate();
}
