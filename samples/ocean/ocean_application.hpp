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

#ifndef ACPP_SAMPLES_OCEAN_APPLICATION_HPP
#define ACPP_SAMPLES_OCEAN_APPLICATION_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "ocean_simulation.hpp"

struct application_options {
  std::uint32_t window_width = 1280;
  std::uint32_t window_height = 720;
  /// Edge length of the simulation grid, must be a power of two.
  std::uint32_t texture_size = 256;
  /// Number of vertices per edge of the rendered ocean patch.
  std::uint32_t grid_size = 256;
  /// Distance between two neighboring grid vertices in meters.
  float mesh_spacing = 2.0f;
  /// Prefer VK_PRESENT_MODE_IMMEDIATE_KHR (no vsync).
  bool immediate = false;
  /// Enable the Vulkan validation layers.
  bool validation_layers = false;
};

/// Ocean surface simulation sample.
///
/// The compute part of the demo (the FFT based wave simulation) is performed
/// with SYCL, the results are uploaded into Vulkan textures which are sampled
/// by the vertex and fragment shaders when rendering the ocean grid.
class ocean_application {
public:
  explicit ocean_application(const application_options &opts);
  ~ocean_application();

  ocean_application(const ocean_application &) = delete;
  ocean_application &operator=(const ocean_application &) = delete;

  void run();

private:
  struct vertex {
    glm::vec3 position;
    glm::vec2 tex_coord;
  };

  // Must match the ViewData uniform block of the shaders (std140 layout).
  struct uniform_buffer_object {
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 sun_dir;
    float choppiness;
    float alt_scale;
  };

  struct camera_state {
    glm::vec3 eye{0.0f, -220.0f, 80.0f};
    glm::vec3 dir{0.0f, 1.0f, -0.2f};
    glm::vec3 up{0.0f, 0.0f, 1.0f};
    glm::vec3 rvec{1.0f, 0.0f, 0.0f};
    glm::vec2 begin{0.0f, 0.0f};
    float yaw = 0.0f;
    float pitch = 0.0f;
    bool drag = false;
  };

  enum interop_texture {
    texture_displacement = 0,
    texture_normal_map,
    texture_count
  };

  struct texture_resources {
    std::vector<VkImage> images;
    std::vector<VkDeviceMemory> memories;
    std::vector<VkImageView> views;
  };

  // setup
  void init_window();
  void init_vulkan();
  void init_simulation();
  void main_loop();
  void cleanup();

  void create_instance();
  void create_surface();
  void pick_physical_device();
  void create_logical_device();
  void create_swap_chain();
  void create_swap_chain_image_views();
  void create_render_pass();
  void create_descriptor_set_layout();
  void create_graphics_pipelines();
  void create_depth_resources();
  void create_framebuffers();
  void create_command_pool();
  void create_grid_buffers();
  void create_textures();
  void create_texture_sampler();
  void create_uniform_buffers();
  void create_descriptor_pool();
  void create_descriptor_sets();
  void create_command_buffers();
  void create_sync_objects();

  // per frame work
  void draw_frame();
  void update_simulation(std::uint32_t image_index);
  void update_uniforms(std::uint32_t image_index);
  void record_command_buffer(std::uint32_t image_index);
  void update_window_title();

  // input handling
  void on_key(int key);
  void on_mouse_button(int button, int action);
  void on_mouse_move(double x, double y);
  void on_scroll(double offset);

  // helpers
  std::uint32_t find_memory_type(std::uint32_t type_filter,
                                 VkMemoryPropertyFlags properties) const;
  void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkBuffer &buffer,
                     VkDeviceMemory &memory);
  void copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
  void create_image(std::uint32_t width, std::uint32_t height, VkFormat format,
                    VkImageTiling tiling, VkImageUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkImage &image,
                    VkDeviceMemory &memory);
  VkImageView create_image_view(VkImage image, VkFormat format,
                                VkImageAspectFlags aspect_flags);
  void transition_image_layout(VkImage image, VkImageLayout old_layout,
                               VkImageLayout new_layout);
  void copy_buffer_to_image(VkBuffer buffer, VkDeviceSize buffer_offset,
                            VkImage image, std::uint32_t width,
                            std::uint32_t height);
  VkCommandBuffer begin_single_time_commands();
  void end_single_time_commands(VkCommandBuffer command_buffer);
  VkShaderModule create_shader_module(const std::vector<char> &code);
  VkFormat find_depth_format() const;

  application_options _opts;
  std::string _app_name = "AdaptiveCpp ocean surface simulation";

  GLFWwindow *_window = nullptr;
  camera_state _camera;

  std::unique_ptr<ocean_simulation> _simulation;

  // ocean parameters
  ocean_parameters _params;
  bool _params_changed = true;
  float _choppiness = 10.0f;
  float _alt_scale = 20.0f;
  bool _animate = true;
  bool _wireframe_mode = false;
  bool _show_fps = true;

  std::chrono::steady_clock::time_point _start =
      std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point _fps_last_time =
      std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point _last_update =
      std::chrono::steady_clock::now();
  float _elapsed = 0.0f;
  int _delta_frames = 0;

  // Vulkan objects
  VkInstance _instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT _debug_messenger = VK_NULL_HANDLE;
  VkSurfaceKHR _surface = VK_NULL_HANDLE;
  VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
  VkDevice _device = VK_NULL_HANDLE;
  VkQueue _graphics_queue = VK_NULL_HANDLE;
  VkQueue _present_queue = VK_NULL_HANDLE;
  std::uint32_t _graphics_family = 0;
  std::uint32_t _present_family = 0;
  bool _fill_mode_non_solid = false;

  VkSwapchainKHR _swap_chain = VK_NULL_HANDLE;
  std::vector<VkImage> _swap_chain_images;
  VkFormat _swap_chain_image_format = VK_FORMAT_UNDEFINED;
  VkExtent2D _swap_chain_extent{};
  std::vector<VkImageView> _swap_chain_image_views;
  std::vector<VkFramebuffer> _swap_chain_framebuffers;

  VkImage _depth_image = VK_NULL_HANDLE;
  VkDeviceMemory _depth_image_memory = VK_NULL_HANDLE;
  VkImageView _depth_image_view = VK_NULL_HANDLE;

  VkRenderPass _render_pass = VK_NULL_HANDLE;
  VkDescriptorSetLayout _descriptor_set_layout = VK_NULL_HANDLE;
  VkPipelineLayout _pipeline_layout = VK_NULL_HANDLE;
  VkPipeline _graphics_pipeline = VK_NULL_HANDLE;
  VkPipeline _wireframe_pipeline = VK_NULL_HANDLE;

  VkCommandPool _command_pool = VK_NULL_HANDLE;

  // staging buffer used to upload the SYCL simulation results
  VkBuffer _staging_buffer = VK_NULL_HANDLE;
  VkDeviceMemory _staging_buffer_memory = VK_NULL_HANDLE;
  void *_staging_buffer_mapped = nullptr;

  std::array<texture_resources, texture_count> _textures;
  std::array<VkSampler, texture_count> _texture_samplers{};

  std::vector<vertex> _grid_vertices;
  std::vector<std::uint32_t> _grid_indices;
  VkBuffer _vertex_buffer = VK_NULL_HANDLE;
  VkDeviceMemory _vertex_buffer_memory = VK_NULL_HANDLE;
  VkBuffer _index_buffer = VK_NULL_HANDLE;
  VkDeviceMemory _index_buffer_memory = VK_NULL_HANDLE;

  std::vector<VkBuffer> _uniform_buffers;
  std::vector<VkDeviceMemory> _uniform_buffer_memories;
  std::vector<void *> _uniform_buffers_mapped;

  VkDescriptorPool _descriptor_pool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> _descriptor_sets;
  std::vector<VkCommandBuffer> _command_buffers;

  std::vector<VkSemaphore> _image_available_semaphores;
  std::vector<VkSemaphore> _render_finished_semaphores;
  std::vector<VkFence> _in_flight_fences;
  std::vector<VkFence> _images_in_flight;
  std::size_t _current_frame = 0;
};

#endif
