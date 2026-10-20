# AdaptiveCpp samples

Standalone samples that are built against an *installed* AdaptiveCpp. They are
not part of the AdaptiveCpp build itself, so that they can pull in third-party
dependencies (windowing, graphics APIs, ...) without affecting the main build.

## Building

```bash
cmake -DACPP_SOURCE_ROOT=<path to the AdaptiveCpp sources> \
      -DAdaptiveCpp_DIR=<install prefix>/lib/cmake/AdaptiveCpp \
      -DACPP_TARGETS=omp \
      -S samples -B build-samples
cmake --build build-samples
```

Samples whose dependencies are missing are skipped with a warning.

## sycl_on_vulkan

A Vulkan application -- window, swapchain, present loop, free-fly camera --
whose frames are rendered by a SYCL kernel instead of a Vulkan compute shader.
It is the SYCL counterpart of the Khronos Vulkan tutorial sample
[`05_opencl_on_vulkan.cpp`](https://github.com/KhronosGroup/Vulkan-Tutorial/blob/main/attachments/compute/05_opencl_on_vulkan.cpp),
which drives a Vulkan presentation loop with an OpenCL C kernel.

The scene is a raymarched, instanced forest: a single tree signed distance
field repeated over an infinite grid with per-instance variation. The kernel is
ordinary C++, so AdaptiveCpp compiles it for whatever backend you built for
(OpenMP, CUDA, HIP, Level Zero, ...).

Requirements: the Vulkan loader and headers, GLFW3, and an installed
AdaptiveCpp. A Vulkan swapchain-capable GPU (or a software implementation such
as lavapipe) is needed to present.

### Interop paths

Vulkan owns the buffer that is presented; SYCL has to write it. Two paths are
supported, picked automatically and overridable with `--interop=`:

* `zero-copy` -- used when the SYCL device reports
  `sycl::aspect::usm_system_allocations` (the AdaptiveCpp OpenMP backend, and
  host-unified devices with system USM support). The kernel writes *directly*
  into the host-visible, host-coherent Vulkan buffer that is presented, so no
  staging or copying takes place at all.
* `copy` -- used otherwise (a discrete GPU, for example). The frame is rendered
  into a device USM allocation and copied once per frame into the Vulkan
  buffer. This mirrors the per-frame readback that the OpenCL sample needs for
  its clvk runtime path.

A true zero-copy path for a discrete GPU would require importing the Vulkan
allocation through an external-memory extension on both sides; SYCL does not
expose a portable mechanism for that, which is exactly the limitation that the
OpenCL sample runs into as well.

### Running

```bash
./build-samples/sycl_on_vulkan/sycl_on_vulkan
```

Options:

| Option | Meaning |
| --- | --- |
| `--device=cpu\|gpu\|default` | which SYCL device to render on |
| `--interop=zero-copy\|copy` | force an interop path instead of auto-detecting |
| `--list-devices` | print the available SYCL devices and whether they support system USM |

Controls: `W`/`A`/`S`/`D` move, `E`/`Q` up/down, drag with the left mouse button
to look around, `Shift` to boost, `R` to reset the camera, `ESC` to quit.
