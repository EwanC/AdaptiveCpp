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

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <iostream>
#include <string>

#include "ocean_application.hpp"

namespace {

bool is_power_of_two(std::uint32_t value) {
  return value >= 2 && (value & (value - 1)) == 0;
}

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "\n"
      << "  --window_width <n>    initial window width (default 1280)\n"
      << "  --window_height <n>   initial window height (default 720)\n"
      << "  --texture_size <n>    simulation resolution, power of two "
         "(default 256)\n"
      << "  --grid_size <n>       vertices per edge of the ocean patch "
         "(default 256)\n"
      << "  --mesh_spacing <f>    distance between grid vertices in meters "
         "(default 2.0)\n"
      << "  --immediate           prefer VK_PRESENT_MODE_IMMEDIATE_KHR (no "
         "vsync)\n"
      << "  --validation_layers   enable the Vulkan validation layers\n"
      << "  --help                print this message\n";
}

bool parse_command_line(int argc, char **argv, application_options &opts) {
  auto value_of = [argc, argv](int &i) -> std::string {
    if (i + 1 >= argc)
      throw std::runtime_error{std::string{"Missing value for option "} +
                               argv[i]};
    return argv[++i];
  };

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return false;
    } else if (arg == "--window_width") {
      opts.window_width = std::stoul(value_of(i));
    } else if (arg == "--window_height") {
      opts.window_height = std::stoul(value_of(i));
    } else if (arg == "--texture_size") {
      opts.texture_size = std::stoul(value_of(i));
    } else if (arg == "--grid_size") {
      opts.grid_size = std::stoul(value_of(i));
    } else if (arg == "--mesh_spacing") {
      opts.mesh_spacing = std::stof(value_of(i));
    } else if (arg == "--immediate") {
      opts.immediate = true;
    } else if (arg == "--validation_layers") {
      opts.validation_layers = true;
    } else {
      throw std::runtime_error{"Unknown command line option: " + arg};
    }
  }

  if (!is_power_of_two(opts.texture_size))
    throw std::runtime_error{"--texture_size must be a power of two"};
  if (opts.grid_size < 2)
    throw std::runtime_error{"--grid_size must be at least 2"};
  if (opts.window_width == 0 || opts.window_height == 0)
    throw std::runtime_error{"window dimensions must not be zero"};

  return true;
}

} // namespace

int main(int argc, char **argv) {
  application_options opts;

  try {
    if (!parse_command_line(argc, argv, opts))
      return EXIT_SUCCESS;

    ocean_application app{opts};
    app.run();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
