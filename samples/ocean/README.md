# Ocean surface simulation with SYCL and Vulkan

This sample is a SYCL version of the [ocean sample](https://github.com/KhronosGroup/OpenCL-SDK/pull/110)
of the Khronos OpenCL-SDK: the compute part of the demo, which in the original
sample is written in OpenCL, is implemented in SYCL here, while rendering is
still performed with Vulkan.

## Sample purpose

The sample simulates an ocean surface with the FFT based approach described in
[Realtime GPGPU FFT ocean water simulation](https://tore.tuhh.de/bitstream/11420/1439/1/GPGPU_FFT_Ocean_Simulation.pdf).
Every frame, SYCL kernels compute a displacement map and a normal map of the
water surface. Both maps are uploaded into Vulkan textures, which the vertex
and fragment shaders sample when rendering a flat grid of vertices, turning it
into an animated, lit ocean surface.

The sample shows

* how a non-trivial, real-time physical simulation can be expressed with SYCL
  USM allocations and `parallel_for`,
* how a SYCL simulation can be combined with a graphics API in a single
  application, and
* how to keep the SYCL part of an application separate from the rest of the
  code: only `ocean_simulation.cpp` is compiled by the SYCL compiler, all
  Vulkan code is ordinary C++.

Unlike the OpenCL sample, this sample does not share device memory between the
compute and the graphics API: SYCL does not offer a vendor-neutral external
memory interoperability mechanism, so the simulation results are copied through
a host visible staging buffer, which corresponds to the fallback path of the
OpenCL sample when `cl_khr_external_memory` is unavailable.

## Dependencies

* an AdaptiveCpp installation
* the Vulkan loader and headers
* GLFW 3.3 or newer
* GLM
* `glslc` or `glslangValidator` to compile the shaders to SPIR-V

If any of these cannot be found, the sample is skipped during CMake
configuration.

## Building

```
cd samples
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/adaptivecpp/install
cmake --build build
./build/ocean/ocean
```

The compiled shaders are placed in a `shaders` directory next to the
executable, where the application looks for them at runtime.

## Application flow

During startup the application

* creates a GLFW window and a Vulkan instance, device, swap chain, render pass
  and graphics pipelines,
* creates the ocean grid vertex/index buffers and one displacement and normal
  map texture per swap chain image, and
* creates the SYCL simulation, which allocates its device memory and computes
  the FFT twiddle factors and the random noise texture.

Each frame then

* recomputes the initial wave spectrum if the ocean parameters changed,
* runs the simulation kernels for the current simulation time,
* copies the resulting displacement and normal maps into the Vulkan textures
  of the current swap chain image, and
* records and submits the command buffer that draws the ocean grid.

### Command line options

```
--window_width <n>    initial window width (default 1280)
--window_height <n>   initial window height (default 720)
--texture_size <n>    simulation resolution, power of two (default 256)
--grid_size <n>       vertices per edge of the ocean patch (default 256)
--mesh_spacing <f>    distance between grid vertices in meters (default 2.0)
--immediate           prefer VK_PRESENT_MODE_IMMEDIATE_KHR (no vsync)
--validation_layers   enable the Vulkan validation layers
```

### Controls

The ocean parameters can be modified while the simulation is running:

* `a`/`z` - increase/decrease wind magnitude
* `s`/`x` - change wind heading
* `d`/`c` - increase/decrease waving amplitude
* `f`/`v` - increase/decrease wave choppiness
* `g`/`b` - increase/decrease additional altitude scale

In addition, `space` pauses the simulation, `w` toggles wireframe rendering,
`e` toggles the frame rate display in the window title and `escape` closes the
window. Dragging with the left mouse button rotates the camera, the mouse wheel
moves it forwards and backwards.

## Kernel logic

The simulation consists of the following SYCL kernels, which are direct ports
of the OpenCL kernels of the original sample:

| Kernel | Description |
| ------ | ----------- |
| twiddle factors | precomputes the butterfly indices and twiddle factors of the FFT (once) |
| initial spectrum | computes the time independent Phillips spectrum `h0(k)` from a noise texture |
| time spectrum | evolves the spectrum to the current point in time and derives the horizontal displacement spectra |
| FFT | butterfly passes of a 2D inverse FFT, first in horizontal, then in vertical direction |
| inversion | combines and scales the three transformed components into the displacement map |
| normals | computes surface normals and a foam factor from the displacement map |

Images are not used by the simulation; the maps are stored in USM allocations
and the texture filtering that the OpenCL kernels obtain from samplers is
implemented explicitly where it is needed.

## Used SYCL API surface

```c++
sycl::queue::queue(const sycl::property_list&)
sycl::queue::get_device()
sycl::queue::parallel_for(sycl::range<2>, Kernel)
sycl::queue::memcpy(void*, const void*, size_t)
sycl::queue::wait()
sycl::device::get_info<sycl::info::device::name>()
sycl::malloc_device<T>(size_t, sycl::queue&)
sycl::free(void*, sycl::queue&)
sycl::item<2>::operator[](int)
sycl::vec<float, N>
sycl::length(), sycl::normalize(), sycl::dot(), sycl::clamp()
sycl::sqrt(), sycl::exp(), sycl::log(), sycl::sin(), sycl::cos(),
sycl::fmod(), sycl::pown(), sycl::floor()
```
