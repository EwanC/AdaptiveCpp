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

#ifndef ACPP_SAMPLES_OCEAN_SIMULATION_HPP
#define ACPP_SAMPLES_OCEAN_SIMULATION_HPP

#include <cstddef>
#include <memory>
#include <string>

/// Ocean surface parameters that can be modified at runtime.
///
/// Changing any of the members requires recalculation of the initial
/// (time independent) wave spectrum, which is why they are grouped here.
struct ocean_parameters {
  /// Wind speed in m/s.
  float wind_magnitude = 30.0f;
  /// Wind heading in degrees.
  float wind_angle = 45.0f;
  /// Phillips spectrum amplitude.
  float amplitude = 80.0f;
  /// Factor suppressing very small (capillary) waves.
  float supress_factor = 0.1f;

  bool operator==(const ocean_parameters &other) const {
    return wind_magnitude == other.wind_magnitude &&
           wind_angle == other.wind_angle && amplitude == other.amplitude &&
           supress_factor == other.supress_factor;
  }
  bool operator!=(const ocean_parameters &other) const {
    return !(*this == other);
  }
};

/// SYCL implementation of a FFT based ocean surface simulation.
///
/// The simulation follows the approach described in
/// "Realtime GPGPU FFT ocean water simulation" and is a port of the OpenCL
/// kernels of the OpenCL-SDK ocean sample to SYCL.
///
/// All computations are carried out on the SYCL device; only the final
/// displacement and normal maps are copied back to the host so that they can
/// be uploaded into Vulkan textures for rendering.
///
/// The SYCL types are hidden behind a pimpl so that the rendering code does
/// not have to be compiled by the SYCL compiler.
class ocean_simulation {
public:
  /// \param texture_size edge length of the (square) simulation grid. Must be
  ///        a power of two.
  /// \param patch_size edge length of the simulated ocean patch in meters.
  ocean_simulation(std::size_t texture_size, float patch_size);
  ~ocean_simulation();

  ocean_simulation(const ocean_simulation &) = delete;
  ocean_simulation &operator=(const ocean_simulation &) = delete;

  /// Name of the SYCL device the simulation runs on.
  const std::string &device_name() const;

  std::size_t texture_size() const;
  /// Number of bytes of a displacement or normal map.
  std::size_t map_size_in_bytes() const;

  /// Recalculate the time independent part of the wave spectrum. Needs to be
  /// called whenever the ocean parameters change.
  void set_parameters(const ocean_parameters &params);

  /// Advance the simulation to the given point in time (in seconds).
  void update(float time);

  /// Copy the displacement map (RGBA32F, texture_size x texture_size) into
  /// \c dst, which must provide \c map_size_in_bytes() bytes.
  void copy_displacement_map(void *dst);
  /// Copy the normal map (RGBA32F, texture_size x texture_size) into \c dst,
  /// which must provide \c map_size_in_bytes() bytes.
  void copy_normal_map(void *dst);

private:
  class impl;
  std::unique_ptr<impl> _impl;
};

#endif
