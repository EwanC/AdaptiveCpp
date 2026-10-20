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

// The simulation implemented here is a SYCL port of the OpenCL kernels of the
// ocean sample of the Khronos OpenCL-SDK. It follows the algorithm described
// in "Realtime GPGPU FFT ocean water simulation" (Fynn-Jorin Flügge).

#include "ocean_simulation.hpp"

#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

namespace {

using complex = sycl::vec<float, 2>;
using float2 = sycl::vec<float, 2>;
using float4 = sycl::vec<float, 4>;

constexpr float pi = 3.14159265359f;
constexpr float gravity = 9.81f;

inline complex cmul(complex a, complex b) {
  return complex{a.x() * b.x() - a.y() * b.y(), a.x() * b.y() + a.y() * b.x()};
}

inline complex cadd(complex a, complex b) {
  return complex{a.x() + b.x(), a.y() + b.y()};
}

inline complex cconj(complex a) { return complex{a.x(), -a.y()}; }

inline float4 gauss_rnd(float4 rnd) {
  const float u0 = 2.0f * pi * rnd.x();
  const float v0 = sycl::sqrt(-2.0f * sycl::log(rnd.y()));
  const float u1 = 2.0f * pi * rnd.z();
  const float v1 = sycl::sqrt(-2.0f * sycl::log(rnd.w()));

  return float4{v0 * sycl::cos(u0), v0 * sycl::sin(u0), v1 * sycl::cos(u1),
                v1 * sycl::sin(u1)};
}

inline int wrap(int v, int n) {
  const int r = v % n;
  return (r < 0) ? r + n : r;
}

// Emulates an image sampler using normalized coordinates, linear filtering
// and repeat addressing.
inline float4 sample_linear_repeat(const float4 *img, int n, float u,
                                   float v) {
  const float fx = u * static_cast<float>(n) - 0.5f;
  const float fy = v * static_cast<float>(n) - 0.5f;

  const float x0f = sycl::floor(fx);
  const float y0f = sycl::floor(fy);

  const float tx = fx - x0f;
  const float ty = fy - y0f;

  const int x0 = wrap(static_cast<int>(x0f), n);
  const int y0 = wrap(static_cast<int>(y0f), n);
  const int x1 = wrap(x0 + 1, n);
  const int y1 = wrap(y0 + 1, n);

  const float4 c00 = img[y0 * n + x0];
  const float4 c10 = img[y0 * n + x1];
  const float4 c01 = img[y1 * n + x0];
  const float4 c11 = img[y1 * n + x1];

  const float4 top = c00 + (c10 - c00) * tx;
  const float4 bottom = c01 + (c11 - c01) * tx;

  return top + (bottom - top) * ty;
}

std::size_t log2_of(std::size_t n) {
  std::size_t result = 0;
  while ((std::size_t{1} << result) < n)
    ++result;
  return result;
}

// Bit reversed indices needed by the first stage of the FFT.
std::vector<int> make_bit_reversed_indices(std::size_t n) {
  const std::size_t bits = log2_of(n);
  std::vector<int> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t reversed = 0;
    for (std::size_t b = 0; b < bits; ++b)
      if (i & (std::size_t{1} << b))
        reversed |= std::size_t{1} << (bits - 1 - b);
    result[i] = static_cast<int>(reversed);
  }
  return result;
}

} // namespace

class ocean_simulation::impl {
public:
  impl(std::size_t texture_size, float patch_size)
      : _queue{sycl::property_list{sycl::property::queue::in_order{}}},
        _n{texture_size}, _patch_size{patch_size},
        _log2n{log2_of(texture_size)} {

    if (texture_size < 2 || (texture_size & (texture_size - 1)) != 0)
      throw std::invalid_argument{
          "ocean_simulation: texture size must be a power of two"};

    _device_name = _queue.get_device().get_info<sycl::info::device::name>();

    const std::size_t num_texels = _n * _n;

    _noise = sycl::malloc_device<float4>(num_texels, _queue);
    _h0k = sycl::malloc_device<float4>(num_texels, _queue);
    _twiddle = sycl::malloc_device<float4>(_log2n * _n, _queue);
    _displacement = sycl::malloc_device<float4>(num_texels, _queue);
    _normals = sycl::malloc_device<float4>(num_texels, _queue);
    for (int i = 0; i < 3; ++i) {
      _dxyz[i] = sycl::malloc_device<complex>(num_texels, _queue);
      _pong[i] = sycl::malloc_device<complex>(num_texels, _queue);
    }

    if (!_noise || !_h0k || !_twiddle || !_displacement || !_normals)
      throw std::runtime_error{
          "ocean_simulation: could not allocate device memory"};
    for (int i = 0; i < 3; ++i)
      if (!_dxyz[i] || !_pong[i])
        throw std::runtime_error{
            "ocean_simulation: could not allocate device memory"};

    init_noise();
    init_twiddle_factors();
  }

  ~impl() {
    sycl::free(_noise, _queue);
    sycl::free(_h0k, _queue);
    sycl::free(_twiddle, _queue);
    sycl::free(_displacement, _queue);
    sycl::free(_normals, _queue);
    for (int i = 0; i < 3; ++i) {
      sycl::free(_dxyz[i], _queue);
      sycl::free(_pong[i], _queue);
    }
  }

  const std::string &device_name() const { return _device_name; }
  std::size_t texture_size() const { return _n; }
  std::size_t map_size_in_bytes() const { return _n * _n * sizeof(float4); }

  void set_parameters(const ocean_parameters &params) {
    _params = params;
    init_spectrum();
  }

  void update(float time) {
    time_spectrum(time);
    for (int component = 0; component < 3; ++component)
      inverse_fft(component);
    inversion();
    normals();
  }

  void copy_displacement_map(void *dst) {
    _queue.memcpy(dst, _displacement, map_size_in_bytes()).wait();
  }

  void copy_normal_map(void *dst) {
    _queue.memcpy(dst, _normals, map_size_in_bytes()).wait();
  }

private:
  void init_noise() {
    std::vector<float4> noise(_n * _n);

    std::random_device dev;
    std::mt19937 rng{dev()};
    std::uniform_real_distribution<float> dist{0.0f, 1.0f};

    for (auto &texel : noise)
      texel = float4{dist(rng), dist(rng), dist(rng), dist(rng)};

    _queue.memcpy(_noise, noise.data(), noise.size() * sizeof(float4)).wait();
  }

  void init_twiddle_factors() {
    const std::vector<int> bit_reversed_host = make_bit_reversed_indices(_n);

    int *bit_reversed = sycl::malloc_device<int>(_n, _queue);
    if (!bit_reversed)
      throw std::runtime_error{
          "ocean_simulation: could not allocate device memory"};
    _queue.memcpy(bit_reversed, bit_reversed_host.data(),
                  bit_reversed_host.size() * sizeof(int));

    const int res = static_cast<int>(_n);
    const std::size_t log2n = _log2n;
    float4 *twiddle = _twiddle;

    _queue.parallel_for(sycl::range<2>{_n, log2n}, [=](sycl::item<2> item) {
      const int y = static_cast<int>(item[0]);
      const int x = static_cast<int>(item[1]);

      const float span = sycl::pown(2.0f, x);
      const float k = sycl::fmod(static_cast<float>(y) *
                                     (static_cast<float>(res) / (2.0f * span)),
                                 static_cast<float>(res));
      const complex factor{
          sycl::cos(2.0f * pi * k / static_cast<float>(res)),
          sycl::sin(2.0f * pi * k / static_cast<float>(res))};

      const int butterfly_span = static_cast<int>(span);
      const bool top_wing =
          sycl::fmod(static_cast<float>(y), 2.0f * span) < span;

      float first = 0.0f;
      float second = 0.0f;
      if (x == 0) {
        // first stage operates on bit reversed indices
        const int index = top_wing ? y : y - 1;
        first = static_cast<float>(bit_reversed[index]);
        second = static_cast<float>(bit_reversed[index + 1]);
      } else {
        first = static_cast<float>(top_wing ? y : y - butterfly_span);
        second = static_cast<float>(top_wing ? y + butterfly_span : y);
      }

      twiddle[y * log2n + x] =
          float4{factor.x(), factor.y(), first, second};
    });

    _queue.wait();
    sycl::free(bit_reversed, _queue);
  }

  void init_spectrum() {
    const int res = static_cast<int>(_n);
    const float patch_size = _patch_size;
    const std::size_t n = _n;

    const float wind_x =
        _params.wind_magnitude * std::cos(_params.wind_angle * pi / 180.0f);
    const float wind_y =
        _params.wind_magnitude * std::sin(_params.wind_angle * pi / 180.0f);
    const float4 params{wind_x, wind_y, _params.amplitude,
                        _params.supress_factor};

    const float4 *noise = _noise;
    float4 *h0k = _h0k;

    _queue.parallel_for(sycl::range<2>{_n, _n}, [=](sycl::item<2> item) {
      const std::size_t y = item[0];
      const std::size_t x = item[1];

      const float2 fuv{static_cast<float>(x) -
                           static_cast<float>(res - 1) / 2.0f,
                       static_cast<float>(y) -
                           static_cast<float>(res - 1) / 2.0f};
      const float2 k = (2.0f * pi * fuv) / patch_size;
      float k_mag = sycl::length(k);
      if (k_mag < 0.00001f)
        k_mag = 0.00001f;

      const float wind_speed = sycl::length(float2{params.x(), params.y()});
      const float2 wind_dir{params.x() / wind_speed, params.y() / wind_speed};
      const float l_phillips = (wind_speed * wind_speed) / gravity;

      const float4 rnd = sycl::clamp(noise[y * n + x], float4{0.001f},
                                     float4{1.0f});

      const float mag_sq = k_mag * k_mag;
      const float common = sycl::exp(-(1.0f / (mag_sq * l_phillips *
                                               l_phillips))) *
                           sycl::exp(-mag_sq * params.w() * params.w());

      const float2 k_norm = sycl::normalize(k);
      const float dir_k = sycl::dot(k_norm, wind_dir);
      const float dir_minus_k = sycl::dot(-k_norm, wind_dir);

      const float h0k_amp = sycl::sqrt((params.z() / (mag_sq * mag_sq)) *
                                       dir_k * dir_k * common) /
                            sycl::sqrt(2.0f);
      const float h0minusk_amp =
          sycl::sqrt((params.z() / (mag_sq * mag_sq)) * dir_minus_k *
                     dir_minus_k * common) /
          sycl::sqrt(2.0f);

      const float4 rand = gauss_rnd(rnd);
      h0k[y * n + x] = float4{rand.x() * h0k_amp, rand.y() * h0k_amp,
                              rand.z() * h0minusk_amp,
                              rand.w() * h0minusk_amp};
    });
  }

  void time_spectrum(float time) {
    const int res = static_cast<int>(_n);
    const std::size_t n = _n;
    const float patch_size = _patch_size;

    const float4 *h0k = _h0k;
    complex *dx = _dxyz[0];
    complex *dy = _dxyz[1];
    complex *dz = _dxyz[2];

    _queue.parallel_for(sycl::range<2>{_n, _n}, [=](sycl::item<2> item) {
      const std::size_t y = item[0];
      const std::size_t x = item[1];
      const std::size_t index = y * n + x;

      const float2 wave_vec{static_cast<float>(x) -
                                static_cast<float>(res - 1) / 2.0f,
                            static_cast<float>(y) -
                                static_cast<float>(res - 1) / 2.0f};
      const float2 k = (2.0f * pi * wave_vec) / patch_size;
      float k_mag = sycl::length(k);
      if (k_mag < 0.00001f)
        k_mag = 0.00001f;

      const float w = sycl::sqrt(gravity * k_mag);

      const float4 amplitudes = h0k[index];
      const complex fourier_amp{amplitudes.x(), amplitudes.y()};
      const complex fourier_amp_conj =
          cconj(complex{amplitudes.z(), amplitudes.w()});

      const float cos_wt = sycl::cos(w * time);
      const float sin_wt = sycl::sin(w * time);

      // Euler's formula
      const complex exp_iwt{cos_wt, sin_wt};
      const complex exp_iwt_inv{cos_wt, -sin_wt};

      // height
      const complex h_dy =
          cadd(cmul(fourier_amp, exp_iwt), cmul(fourier_amp_conj, exp_iwt_inv));
      // horizontal displacement (choppiness)
      const complex h_dx = cmul(complex{0.0f, k.x() / k_mag}, h_dy);
      const complex h_dz = cmul(complex{0.0f, k.y() / k_mag}, h_dy);

      dx[index] = h_dx;
      dy[index] = h_dy;
      dz[index] = h_dz;
    });
  }

  // 2D inverse FFT of one displacement component, implemented as a sequence
  // of butterfly passes in horizontal and then vertical direction.
  void inverse_fft(int component) {
    complex *ping = _dxyz[component];
    complex *pong = _pong[component];

    bool result_in_pong = false;
    for (int direction = 0; direction < 2; ++direction) {
      for (std::size_t stage = 0; stage < _log2n; ++stage) {
        const complex *src = result_in_pong ? pong : ping;
        complex *dst = result_in_pong ? ping : pong;

        fft_stage(src, dst, direction, static_cast<int>(stage));
        result_in_pong = !result_in_pong;
      }
    }

    // make sure the result always ends up in the ping buffer
    if (result_in_pong)
      _queue.memcpy(ping, pong, _n * _n * sizeof(complex));
  }

  void fft_stage(const complex *src, complex *dst, int direction, int stage) {
    const std::size_t n = _n;
    const std::size_t log2n = _log2n;
    const float4 *twiddle = _twiddle;
    // direction == 0: horizontal pass, direction == 1: vertical pass
    const bool vertical = direction == 1;

    _queue.parallel_for(sycl::range<2>{_n, _n}, [=](sycl::item<2> item) {
      const std::size_t y = item[0];
      const std::size_t x = item[1];

      const std::size_t twiddle_row = vertical ? y : x;
      const float4 data = twiddle[twiddle_row * log2n + stage];

      const std::size_t index0 = static_cast<std::size_t>(data.z());
      const std::size_t index1 = static_cast<std::size_t>(data.w());

      const complex p = vertical ? src[index0 * n + x] : src[y * n + index0];
      const complex q = vertical ? src[index1 * n + x] : src[y * n + index1];

      const complex w{data.x(), data.y()};

      // butterfly operation
      dst[y * n + x] = cadd(p, cmul(w, q));
    });
  }

  void inversion() {
    const std::size_t n = _n;
    const float scale = 1.0f / static_cast<float>(_n * _n);

    const complex *dx = _dxyz[0];
    const complex *dy = _dxyz[1];
    const complex *dz = _dxyz[2];
    float4 *displacement = _displacement;

    _queue.parallel_for(sycl::range<2>{_n, _n}, [=](sycl::item<2> item) {
      const std::size_t index = item[0] * n + item[1];

      displacement[index] = float4{dx[index].x() * scale, dy[index].x() * scale,
                                   dz[index].x() * scale, 1.0f};
    });
  }

  void normals() {
    const int res = static_cast<int>(_n);
    const std::size_t n = _n;
    constexpr float normal_scale_fac = 3.0f;

    const float4 *displacement = _displacement;
    const float4 *noise = _noise;
    float4 *normals = _normals;

    _queue.parallel_for(sycl::range<2>{_n, _n}, [=](sycl::item<2> item) {
      const std::size_t y = item[0];
      const std::size_t x = item[1];

      const float u = static_cast<float>(x) / static_cast<float>(res);
      const float v = static_cast<float>(y) / static_cast<float>(res);
      const float texel = 1.0f / static_cast<float>(res);

      auto height = [&](float du, float dv) {
        return sample_linear_repeat(displacement, res, u + du, v + dv).y();
      };

      const float dz_c = height(0.0f, 0.0f);
      const float dz_cr = height(texel, 0.0f);
      const float dz_ct = height(0.0f, texel);
      const float dz_cl = height(-texel, 0.0f);
      const float dz_cb = height(0.0f, -texel);
      const float dz_tr = height(texel, texel);
      const float dz_br = height(texel, -texel);
      const float dz_tl = height(-texel, texel);

      const float normal_y =
          dz_c + 2.0f * dz_cb + dz_br - dz_tl - 2.0f * dz_ct - dz_tr;
      const float normal_x =
          dz_c + 2.0f * dz_cl + dz_tl - dz_br - 2.0f * dz_cr - dz_tr;

      const sycl::vec<float, 3> normal = sycl::normalize(
          sycl::vec<float, 3>{normal_x, normal_y, 1.0f / normal_scale_fac});

      // the noise texture provides a pseudo random foam factor
      const float4 foam_noise =
          sample_linear_repeat(noise, res, u * 4.0f, v * 4.0f);
      const int component = static_cast<int>((x + y) % 4);
      const float foam = foam_noise[component];

      normals[y * n + x] =
          float4{normal.x(), normal.y(), normal.z(), foam};
    });
  }

  sycl::queue _queue;
  std::string _device_name;

  std::size_t _n;
  float _patch_size;
  std::size_t _log2n;

  ocean_parameters _params;

  float4 *_noise = nullptr;
  float4 *_h0k = nullptr;
  float4 *_twiddle = nullptr;
  float4 *_displacement = nullptr;
  float4 *_normals = nullptr;
  complex *_dxyz[3] = {nullptr, nullptr, nullptr};
  complex *_pong[3] = {nullptr, nullptr, nullptr};
};

ocean_simulation::ocean_simulation(std::size_t texture_size, float patch_size)
    : _impl{std::make_unique<impl>(texture_size, patch_size)} {}

ocean_simulation::~ocean_simulation() = default;

const std::string &ocean_simulation::device_name() const {
  return _impl->device_name();
}

std::size_t ocean_simulation::texture_size() const {
  return _impl->texture_size();
}

std::size_t ocean_simulation::map_size_in_bytes() const {
  return _impl->map_size_in_bytes();
}

void ocean_simulation::set_parameters(const ocean_parameters &params) {
  _impl->set_parameters(params);
}

void ocean_simulation::update(float time) { _impl->update(time); }

void ocean_simulation::copy_displacement_map(void *dst) {
  _impl->copy_displacement_map(dst);
}

void ocean_simulation::copy_normal_map(void *dst) {
  _impl->copy_normal_map(dst);
}
