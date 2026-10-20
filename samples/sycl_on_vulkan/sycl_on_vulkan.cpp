// ===========================================================================
// sycl_on_vulkan.cpp  --  Vulkan rendering powered by a SYCL kernel
// ===========================================================================
// A normal *Vulkan* application -- window, swapchain, present loop, free-fly
// camera -- that draws its scene with a SYCL kernel compiled by AdaptiveCpp
// instead of a hand-written Vulkan compute shader. It is the SYCL counterpart
// of the Khronos "OpenCL on Vulkan" tutorial sample: a Vulkan engine reusing a
// portable compute kernel written in plain C++.
//
// The scene is a raymarched, instanced forest: one tree signed distance field
// repeated across an infinite grid with per-instance variation. You fly through
// it in real time.
//
// Two interop paths exist, selected automatically (or with --interop=):
//
//   * zero-copy: if the SYCL device can access ordinary system memory
//     (sycl::aspect::usm_system_allocations -- true for the AdaptiveCpp OpenMP
//     backend, and for host-unified devices supporting system USM), the kernel
//     writes *directly* into the host-visible Vulkan buffer that is presented.
//     No copies, no staging: the SYCL kernel becomes the renderer of the
//     Vulkan engine.
//
//   * copy: otherwise (a discrete GPU, for instance) the frame is rendered into
//     a device USM allocation and copied once per frame into the Vulkan buffer.
//     This is what the OpenCL sample has to do for its clvk runtime path.
//
// Controls: W/A/S/D move, E/Q up/down, mouse-drag look, Shift boost, R reset,
//           ESC quit.
// Options:  --interop=zero-copy|copy   --device=cpu|gpu|default   --list-devices
// ===========================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <sycl/sycl.hpp>

// ---------------------------------------------------------------------------
constexpr uint32_t kWidth = 1280;
constexpr uint32_t kHeight = 720;
constexpr int kLocal = 16; // work group edge length
constexpr int kAcquireSemas = 3;

const std::vector<const char *> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"};
#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

// Camera / framing information handed to the SYCL kernel by value.
struct Params {
  int32_t width;
  int32_t height;
  float camX, camY, camZ;
  float camYaw, camPitch;
  float fog;
};

static uint32_t roundUp(uint32_t v, uint32_t m) { return (v + m - 1) / m * m; }

// ===========================================================================
// The kernel: a raymarched, instanced forest.
//
// This is ordinary C++ -- the exact same code also runs on the host. AdaptiveCpp
// compiles it for whichever device the queue was created on.
// ===========================================================================
namespace forest {

constexpr int kMaxSteps = 96;
constexpr float kMaxDist = 60.0f;
constexpr float kEpsilon = 1.5e-3f;
constexpr float kCellSize = 6.0f;

struct Vec3 {
  float x, y, z;
};

inline Vec3 operator+(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline Vec3 operator-(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length(Vec3 a) { return sycl::sqrt(dot(a, a)); }
inline Vec3 normalize(Vec3 a) {
  const float l = length(a);
  return l > 0.0f ? a * (1.0f / l) : a;
}

// Cheap deterministic per-cell hash, in [0, 1).
inline float hash2(float ix, float iz) {
  const float h = sycl::sin(ix * 127.1f + iz * 311.7f) * 43758.5453f;
  return h - sycl::floor(h);
}

inline float sdSphere(Vec3 p, float r) { return length(p) - r; }

// Capped cylinder along +y, from y = 0 to y = h.
inline float sdTrunk(Vec3 p, float r, float h) {
  const float dxz = sycl::sqrt(p.x * p.x + p.z * p.z) - r;
  const float dy = sycl::fabs(p.y - 0.5f * h) - 0.5f * h;
  const float ox = sycl::max(dxz, 0.0f);
  const float oy = sycl::max(dy, 0.0f);
  return sycl::min(sycl::max(dxz, dy), 0.0f) + sycl::sqrt(ox * ox + oy * oy);
}

// One tree, placed at the origin of its cell. `v` drives the per-instance
// variation (height, crown size).
inline float sdTree(Vec3 p, float v) {
  const float height = 2.4f + 1.8f * v;
  const float trunk = sdTrunk(p, 0.14f + 0.06f * v, height);
  const float crownR = 0.85f + 0.5f * v;
  float crown = sdSphere(p - Vec3{0.0f, height + crownR * 0.45f, 0.0f}, crownR);
  crown = sycl::min(
      crown, sdSphere(p - Vec3{crownR * 0.45f, height * 0.85f, -crownR * 0.3f},
                      crownR * 0.7f));
  crown = sycl::min(
      crown, sdSphere(p - Vec3{-crownR * 0.4f, height * 0.95f, crownR * 0.35f},
                      crownR * 0.6f));
  return sycl::min(trunk, crown);
}

// Scene distance. `material` is 0 for the ground, 1 for a tree.
inline float sdScene(Vec3 p, int &material) {
  float d = p.y; // ground plane
  int mat = 0;

  // Instance the tree over an infinite grid: only the 3x3 neighbourhood of
  // cells around the sample point can contribute.
  const float cx = sycl::floor(p.x / kCellSize);
  const float cz = sycl::floor(p.z / kCellSize);
  for (int iz = -1; iz <= 1; ++iz) {
    for (int ix = -1; ix <= 1; ++ix) {
      const float gx = cx + static_cast<float>(ix);
      const float gz = cz + static_cast<float>(iz);
      const float v = hash2(gx, gz);
      const float jx = hash2(gx + 17.0f, gz - 7.0f);
      const float jz = hash2(gx - 23.0f, gz + 11.0f);
      const Vec3 local = p - Vec3{(gx + 0.25f + 0.5f * jx) * kCellSize, 0.0f,
                                  (gz + 0.25f + 0.5f * jz) * kCellSize};
      const float dt = sdTree(local, v);
      if (dt < d) {
        d = dt;
        mat = 1;
      }
    }
  }
  material = mat;
  return d;
}

inline Vec3 normalAt(Vec3 p) {
  const float e = 1.0e-3f;
  int ignored = 0;
  const float d = sdScene(p, ignored);
  return normalize(Vec3{sdScene(p + Vec3{e, 0.0f, 0.0f}, ignored) - d,
                        sdScene(p + Vec3{0.0f, e, 0.0f}, ignored) - d,
                        sdScene(p + Vec3{0.0f, 0.0f, e}, ignored) - d});
}

// Pack to the R8G8B8A8Unorm layout expected by the Vulkan buffer-to-image copy.
inline uint32_t packRGBA(Vec3 c) {
  auto to8 = [](float v) {
    return static_cast<uint32_t>(sycl::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
  };
  return to8(c.x) | (to8(c.y) << 8) | (to8(c.z) << 16) | (0xFFu << 24);
}

// Render a single pixel.
inline uint32_t shadePixel(const Params &p, int px, int py) {
  const float aspect =
      static_cast<float>(p.width) / static_cast<float>(p.height);
  const float u =
      (2.0f * (static_cast<float>(px) + 0.5f) / static_cast<float>(p.width) -
       1.0f) *
      aspect;
  const float v = 1.0f - 2.0f * (static_cast<float>(py) + 0.5f) /
                             static_cast<float>(p.height);

  const float cy = sycl::cos(p.camYaw), sy = sycl::sin(p.camYaw);
  const float cp = sycl::cos(p.camPitch), sp = sycl::sin(p.camPitch);
  const Vec3 forward{sy * cp, sp, cy * cp};
  const Vec3 right{cy, 0.0f, -sy};
  const Vec3 up{-sy * sp, cp, -cy * sp};

  const Vec3 origin{p.camX, p.camY, p.camZ};
  const Vec3 dir = normalize(forward + right * u + up * v);

  const Vec3 sun = normalize(Vec3{0.45f, 0.75f, 0.30f});
  const Vec3 skyTop{0.32f, 0.52f, 0.85f};
  const Vec3 skyHorizon{0.78f, 0.84f, 0.92f};
  const float t01 = sycl::clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
  const Vec3 sky = skyHorizon + (skyTop - skyHorizon) * t01;

  float t = 0.0f;
  int material = 0;
  bool hit = false;
  for (int i = 0; i < kMaxSteps && t < kMaxDist; ++i) {
    const float d = sdScene(origin + dir * t, material);
    if (d < kEpsilon) {
      hit = true;
      break;
    }
    t += d;
  }

  if (!hit)
    return packRGBA(sky);

  const Vec3 pos = origin + dir * t;
  const Vec3 n = normalAt(pos);
  const float diffuse = sycl::max(dot(n, sun), 0.0f);
  const float ambient = 0.25f + 0.25f * sycl::max(n.y, 0.0f);

  Vec3 albedo;
  if (material == 0) {
    const float checker =
        (static_cast<int>(sycl::floor(pos.x * 0.5f) +
                          sycl::floor(pos.z * 0.5f)) &
         1)
            ? 0.78f
            : 0.62f;
    albedo = Vec3{0.30f, 0.42f, 0.22f} * checker;
  } else {
    const float foliage = sycl::clamp((pos.y - 1.5f) * 0.6f, 0.0f, 1.0f);
    const Vec3 bark{0.36f, 0.26f, 0.17f};
    const Vec3 leaves{0.18f, 0.48f, 0.19f};
    albedo = bark + (leaves - bark) * foliage;
  }

  Vec3 color = albedo * (ambient + diffuse);
  const float fogAmount = sycl::clamp(1.0f - sycl::exp(-p.fog * t), 0.0f, 1.0f);
  color = color + (sky - color) * fogAmount;
  return packRGBA(color);
}

} // namespace forest

// ===========================================================================
// SYCL backend: owns the queue and, for the copy path, the device allocation.
// ===========================================================================
class SyclRenderer {
public:
  enum class Interop { ZeroCopy, Copy };

  SyclRenderer(const std::string &deviceSelection,
               const std::string &forcedInterop)
      : m_queue(makeDevice(deviceSelection),
                sycl::property_list{sycl::property::queue::in_order{}}) {
    const sycl::device dev = m_queue.get_device();
    const bool systemUsm = dev.has(sycl::aspect::usm_system_allocations);

    if (forcedInterop == "zero-copy") {
      if (!systemUsm)
        throw std::runtime_error(
            "--interop=zero-copy requested, but the selected SYCL device does "
            "not support system USM allocations.");
      m_interop = Interop::ZeroCopy;
    } else if (forcedInterop == "copy") {
      m_interop = Interop::Copy;
    } else {
      m_interop = systemUsm ? Interop::ZeroCopy : Interop::Copy;
    }

    std::cout << "[sycl]    device: " << dev.get_info<sycl::info::device::name>()
              << " ("
              << dev.get_platform().get_info<sycl::info::platform::name>()
              << ")\n";
    std::cout << "[sycl]    interop: "
              << (m_interop == Interop::ZeroCopy
                      ? "zero-copy (the kernel writes the mapped Vulkan buffer "
                        "directly)"
                      : "copy (device USM plus one copy per frame)")
              << '\n';
  }

  ~SyclRenderer() {
    if (m_deviceImage)
      sycl::free(m_deviceImage, m_queue);
  }

  SyclRenderer(const SyclRenderer &) = delete;
  SyclRenderer &operator=(const SyclRenderer &) = delete;

  Interop interop() const { return m_interop; }

  // `mapped` points into host-visible, host-coherent Vulkan device memory.
  void init(void *mapped, size_t bytes) {
    m_mapped = mapped;
    m_bytes = bytes;
    if (m_interop == Interop::Copy) {
      m_deviceImage =
          sycl::malloc_device<uint32_t>(bytes / sizeof(uint32_t), m_queue);
      if (!m_deviceImage)
        throw std::runtime_error("could not allocate the SYCL device image");
    }
  }

  // Render one frame; on return, the mapped Vulkan memory holds the image.
  void render(const Params &params) {
    uint32_t *const out = (m_interop == Interop::ZeroCopy)
                              ? static_cast<uint32_t *>(m_mapped)
                              : m_deviceImage;
    const sycl::range<2> global{roundUp(params.height, kLocal),
                                roundUp(params.width, kLocal)};
    const sycl::range<2> local{kLocal, kLocal};
    const int width = params.width;
    const int height = params.height;

    m_queue.parallel_for(
        sycl::nd_range<2>{global, local}, [=](sycl::nd_item<2> item) {
          const int y = static_cast<int>(item.get_global_id(0));
          const int x = static_cast<int>(item.get_global_id(1));
          if (x >= width || y >= height)
            return;
          out[y * width + x] = forest::shadePixel(params, x, y);
        });

    if (m_interop == Interop::Copy)
      m_queue.memcpy(m_mapped, m_deviceImage, m_bytes);
    m_queue.wait_and_throw();
  }

private:
  static sycl::device makeDevice(const std::string &selection) {
    if (selection == "cpu")
      return sycl::device{sycl::cpu_selector_v};
    if (selection == "gpu")
      return sycl::device{sycl::gpu_selector_v};
    if (selection != "default")
      throw std::runtime_error("unknown --device=" + selection +
                               " (expected cpu, gpu or default)");
    return sycl::device{sycl::default_selector_v};
  }

  sycl::queue m_queue;
  Interop m_interop = Interop::Copy;
  void *m_mapped = nullptr;
  size_t m_bytes = 0;
  uint32_t *m_deviceImage = nullptr;
};

// ===========================================================================
class ForestApp {
public:
  ForestApp(std::string device, std::string interop)
      : m_deviceSelection(std::move(device)),
        m_forcedInterop(std::move(interop)) {}

  void run() {
    initWindow();
    createInstance();
    setupDebug();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createCommandPool();
    createSwapchain();
    createIntermediateImage();
    createSharedBuffer();
    initSycl();
    createSyncAndCmd();
    mainLoop();
    cleanup();
  }

private:
  std::string m_deviceSelection;
  std::string m_forcedInterop;

  // ---- window / camera ----
  GLFWwindow *m_window = nullptr;
  bool m_dragging = false;
  double m_lastMx = 0, m_lastMy = 0;
  float m_camX = 0.0f, m_camY = 3.1f, m_camZ = 0.0f;
  float m_yaw = 0.45f, m_pitch = -0.40f;
  double m_lastTime = 0.0;

  // ---- Vulkan core ----
  vk::raii::Context m_ctx;
  vk::raii::Instance m_instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT m_debug = nullptr;
  vk::raii::SurfaceKHR m_surface = nullptr;
  vk::raii::PhysicalDevice m_phys = nullptr;
  vk::raii::Device m_device = nullptr;
  uint32_t m_queueFamily = ~0u;
  vk::raii::Queue m_queue = nullptr;
  vk::raii::CommandPool m_cmdPool = nullptr;

  // ---- swapchain ----
  vk::raii::SwapchainKHR m_swapchain = nullptr;
  std::vector<vk::Image> m_swapImages;
  vk::SurfaceFormatKHR m_swapFormat{};
  vk::Extent2D m_extent{};

  // ---- intermediate RGBA image (the blit converts RGBA -> BGRA) ----
  vk::raii::Image m_interImg = nullptr;
  vk::raii::DeviceMemory m_interMem = nullptr;

  // ---- shared output buffer: SYCL writes it, Vulkan presents it ----
  vk::raii::Buffer m_sharedBuf = nullptr;
  vk::raii::DeviceMemory m_sharedMem = nullptr;
  void *m_sharedMapped = nullptr;
  vk::DeviceSize m_sharedBytes = 0;

  std::unique_ptr<SyclRenderer> m_sycl;

  // ---- per-frame sync ----
  std::vector<vk::raii::Semaphore> m_imageAvail;
  std::vector<vk::raii::Semaphore> m_renderDone;
  int m_acquireIdx = 0;
  vk::raii::CommandBuffer m_cmd = nullptr;
  vk::raii::Fence m_fence = nullptr;

  // =======================================================================
  void initWindow() {
    if (!glfwInit())
      throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    m_window = glfwCreateWindow(kWidth, kHeight,
                                "SYCL-on-Vulkan Forest  |  WASD+EQ move  "
                                "drag=look  Shift=boost  R=reset  ESC=quit",
                                nullptr, nullptr);
    if (!m_window)
      throw std::runtime_error("glfwCreateWindow failed");
    glfwSetWindowUserPointer(m_window, this);
    glfwSetMouseButtonCallback(m_window, cbMouseButton);
    glfwSetCursorPosCallback(m_window, cbCursorPos);
    glfwSetKeyCallback(m_window, cbKey);
  }

  static void cbMouseButton(GLFWwindow *w, int button, int action, int) {
    auto *a = static_cast<ForestApp *>(glfwGetWindowUserPointer(w));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      a->m_dragging = (action == GLFW_PRESS);
      glfwGetCursorPos(w, &a->m_lastMx, &a->m_lastMy);
    }
  }
  static void cbCursorPos(GLFWwindow *w, double mx, double my) {
    auto *a = static_cast<ForestApp *>(glfwGetWindowUserPointer(w));
    if (a->m_dragging) {
      a->m_yaw -= float(mx - a->m_lastMx) * 0.0040f;
      a->m_pitch -= float(my - a->m_lastMy) * 0.0040f;
      a->m_pitch = std::clamp(a->m_pitch, -1.5f, 1.5f);
    }
    a->m_lastMx = mx;
    a->m_lastMy = my;
  }
  static void cbKey(GLFWwindow *w, int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    auto *a = static_cast<ForestApp *>(glfwGetWindowUserPointer(w));
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(w, GLFW_TRUE);
    if (key == GLFW_KEY_R) {
      a->m_camX = 0.0f;
      a->m_camY = 3.1f;
      a->m_camZ = 0.0f;
      a->m_yaw = 0.45f;
      a->m_pitch = -0.40f;
    }
  }

  void updateCamera() {
    const double now = glfwGetTime();
    const float dt = std::min(float(now - m_lastTime), 0.05f);
    m_lastTime = now;
    float speed = 6.0f * dt;
    if (glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
      speed *= 4.0f;
    const float fx = std::sin(m_yaw), fz = std::cos(m_yaw);
    const float rx = std::cos(m_yaw), rz = -std::sin(m_yaw);
    auto mv = [&](float dx, float dy, float dz) {
      m_camX += dx;
      m_camY += dy;
      m_camZ += dz;
    };
    if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS)
      mv(fx * speed, 0.0f, fz * speed);
    if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS)
      mv(-fx * speed, 0.0f, -fz * speed);
    if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS)
      mv(rx * speed, 0.0f, rz * speed);
    if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS)
      mv(-rx * speed, 0.0f, -rz * speed);
    if (glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS)
      mv(0.0f, speed, 0.0f);
    if (glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS)
      mv(0.0f, -speed, 0.0f);
    m_camY = std::max(m_camY, 0.3f);
  }

  Params currentParams() const {
    Params p{};
    p.width = int32_t(m_extent.width);
    p.height = int32_t(m_extent.height);
    p.camX = m_camX;
    p.camY = m_camY;
    p.camZ = m_camZ;
    p.camYaw = m_yaw;
    p.camPitch = m_pitch;
    p.fog = 0.05f;
    return p;
  }

  // =======================================================================
  void createInstance() {
    const vk::ApplicationInfo appInfo{
        .pApplicationName = "SYCL-on-Vulkan Forest",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "AdaptiveCpp",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3};
    std::vector<const char *> layers;
    if (kEnableValidation)
      layers.assign(kValidationLayers.begin(), kValidationLayers.end());
    uint32_t gc = 0;
    const char **gexts = glfwGetRequiredInstanceExtensions(&gc);
    if (!gexts)
      throw std::runtime_error("glfwGetRequiredInstanceExtensions failed");
    std::vector<const char *> exts(gexts, gexts + gc);
    if (kEnableValidation)
      exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    m_instance = vk::raii::Instance(
        m_ctx, vk::InstanceCreateInfo{
                   .pApplicationInfo = &appInfo,
                   .enabledLayerCount = uint32_t(layers.size()),
                   .ppEnabledLayerNames = layers.data(),
                   .enabledExtensionCount = uint32_t(exts.size()),
                   .ppEnabledExtensionNames = exts.data()});
  }

  void setupDebug() {
    if (!kEnableValidation)
      return;
    m_debug = m_instance.createDebugUtilsMessengerEXT(
        vk::DebugUtilsMessengerCreateInfoEXT{
            .messageSeverity =
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            .pfnUserCallback = &debugCallback});
  }

  void createSurface() {
    VkSurfaceKHR raw = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(*m_instance, m_window, nullptr, &raw) !=
        VK_SUCCESS)
      throw std::runtime_error("glfwCreateWindowSurface failed");
    m_surface = vk::raii::SurfaceKHR(m_instance, raw);
  }

  void pickPhysicalDevice() {
    auto typeScore = [](vk::PhysicalDeviceType t) {
      switch (t) {
      case vk::PhysicalDeviceType::eDiscreteGpu:
        return 4;
      case vk::PhysicalDeviceType::eIntegratedGpu:
        return 3;
      default:
        return 1;
      }
    };
    int best = 0;
    for (auto &pd : m_instance.enumeratePhysicalDevices()) {
      auto qfps = pd.getQueueFamilyProperties();
      uint32_t qf = ~0u;
      for (uint32_t i = 0; i < qfps.size(); ++i)
        if ((qfps[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
            pd.getSurfaceSupportKHR(i, *m_surface)) {
          qf = i;
          break;
        }
      if (qf == ~0u)
        continue;
      auto de = pd.enumerateDeviceExtensionProperties();
      if (!std::any_of(de.begin(), de.end(), [](auto &e) {
            return strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) ==
                   0;
          }))
        continue;
      const int s = typeScore(pd.getProperties().deviceType);
      if (s > best) {
        best = s;
        m_phys = pd;
        m_queueFamily = qf;
      }
    }
    if (!*m_phys)
      throw std::runtime_error("No suitable GPU found");
    std::cout << "[vulkan]  device: " << m_phys.getProperties().deviceName.data()
              << '\n';
  }

  void createLogicalDevice() {
    std::vector<const char *> exts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    vk::StructureChain<vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceVulkan13Features>
        chain = {{}, {.synchronization2 = true}};
    const float prio = 1.0f;
    const vk::DeviceQueueCreateInfo qci{.queueFamilyIndex = m_queueFamily,
                                        .queueCount = 1,
                                        .pQueuePriorities = &prio};
    m_device = vk::raii::Device(
        m_phys, vk::DeviceCreateInfo{
                    .pNext = &chain.get<vk::PhysicalDeviceFeatures2>(),
                    .queueCreateInfoCount = 1,
                    .pQueueCreateInfos = &qci,
                    .enabledExtensionCount = uint32_t(exts.size()),
                    .ppEnabledExtensionNames = exts.data()});
    m_queue = vk::raii::Queue(m_device, m_queueFamily, 0);
  }

  void createCommandPool() {
    m_cmdPool = vk::raii::CommandPool(
        m_device,
        vk::CommandPoolCreateInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = m_queueFamily});
  }

  void createSwapchain() {
    const auto caps = m_phys.getSurfaceCapabilitiesKHR(*m_surface);
    m_extent = (caps.currentExtent.width != UINT32_MAX)
                   ? caps.currentExtent
                   : vk::Extent2D{kWidth, kHeight};
    const auto fmts = m_phys.getSurfaceFormatsKHR(*m_surface);
    m_swapFormat = fmts[0];
    for (const auto &f : fmts)
      if (f.format == vk::Format::eB8G8R8A8Unorm &&
          f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear &&
          !!(m_phys.getFormatProperties(f.format).optimalTilingFeatures &
             vk::FormatFeatureFlagBits::eBlitDst)) {
        m_swapFormat = f;
        break;
      }
    const auto modes = m_phys.getSurfacePresentModesKHR(*m_surface);
    auto mode = vk::PresentModeKHR::eFifo;
    for (auto m : modes)
      if (m == vk::PresentModeKHR::eMailbox)
        mode = m;
    uint32_t imgCount = std::max(3u, caps.minImageCount);
    if (caps.maxImageCount > 0)
      imgCount = std::min(imgCount, caps.maxImageCount);
    m_swapchain = vk::raii::SwapchainKHR(
        m_device, vk::SwapchainCreateInfoKHR{
                      .surface = *m_surface,
                      .minImageCount = imgCount,
                      .imageFormat = m_swapFormat.format,
                      .imageColorSpace = m_swapFormat.colorSpace,
                      .imageExtent = m_extent,
                      .imageArrayLayers = 1,
                      .imageUsage = vk::ImageUsageFlagBits::eTransferDst,
                      .imageSharingMode = vk::SharingMode::eExclusive,
                      .preTransform = caps.currentTransform,
                      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                      .presentMode = mode,
                      .clipped = true});
    m_swapImages = m_swapchain.getImages();
  }

  void createIntermediateImage() {
    m_interImg = vk::raii::Image(
        m_device,
        vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .extent = {m_extent.width, m_extent.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferDst |
                     vk::ImageUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined});
    const auto req = m_interImg.getMemoryRequirements();
    m_interMem = vk::raii::DeviceMemory(
        m_device,
        vk::MemoryAllocateInfo{
            .allocationSize = req.size,
            .memoryTypeIndex =
                findMemoryType(req.memoryTypeBits,
                               vk::MemoryPropertyFlagBits::eDeviceLocal)});
    m_interImg.bindMemory(*m_interMem, 0);
  }

  void createSharedBuffer() {
    m_sharedBytes =
        vk::DeviceSize(m_extent.width) * m_extent.height * sizeof(uint32_t);
    m_sharedBuf = vk::raii::Buffer(
        m_device, vk::BufferCreateInfo{
                      .size = m_sharedBytes,
                      .usage = vk::BufferUsageFlagBits::eTransferSrc,
                      .sharingMode = vk::SharingMode::eExclusive});
    const auto req = m_sharedBuf.getMemoryRequirements();
    m_sharedMem = vk::raii::DeviceMemory(
        m_device,
        vk::MemoryAllocateInfo{
            .allocationSize = req.size,
            .memoryTypeIndex =
                findMemoryType(req.memoryTypeBits,
                               vk::MemoryPropertyFlagBits::eHostVisible |
                                   vk::MemoryPropertyFlagBits::eHostCoherent)});
    m_sharedBuf.bindMemory(*m_sharedMem, 0);
    m_sharedMapped = m_sharedMem.mapMemory(0, m_sharedBytes);
  }

  void initSycl() {
    m_sycl = std::make_unique<SyclRenderer>(m_deviceSelection, m_forcedInterop);
    m_sycl->init(m_sharedMapped, size_t(m_sharedBytes));
  }

  void createSyncAndCmd() {
    for (int i = 0; i < kAcquireSemas; ++i)
      m_imageAvail.emplace_back(m_device, vk::SemaphoreCreateInfo{});
    for (size_t i = 0; i < m_swapImages.size(); ++i)
      m_renderDone.emplace_back(m_device, vk::SemaphoreCreateInfo{});
    m_cmd = std::move(vk::raii::CommandBuffers(
                          m_device,
                          vk::CommandBufferAllocateInfo{
                              .commandPool = *m_cmdPool,
                              .level = vk::CommandBufferLevel::ePrimary,
                              .commandBufferCount = 1})
                          .front());
    m_fence = vk::raii::Fence(
        m_device,
        vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
  }

  // =======================================================================
  void mainLoop() {
    m_lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(m_window)) {
      glfwPollEvents();
      updateCamera();
      drawFrame();
    }
    m_device.waitIdle();
  }

  void drawFrame() {
    if (m_device.waitForFences(*m_fence, vk::True, UINT64_MAX) !=
        vk::Result::eSuccess)
      throw std::runtime_error("vkWaitForFences failed");
    m_device.resetFences(*m_fence);

    // The SYCL kernel produces the frame into the host-visible Vulkan buffer.
    m_sycl->render(currentParams());

    auto &acq = m_imageAvail[m_acquireIdx];
    m_acquireIdx = (m_acquireIdx + 1) % kAcquireSemas;
    uint32_t imageIndex = 0;
    {
      auto [res, idx] = m_swapchain.acquireNextImage(UINT64_MAX, *acq, nullptr);
      if (res == vk::Result::eErrorOutOfDateKHR) {
        m_device.waitIdle();
        return;
      }
      imageIndex = idx;
    }
    recordCommands(imageIndex);

    auto &rd = m_renderDone[imageIndex];
    const vk::PipelineStageFlags waitStage =
        vk::PipelineStageFlagBits::eTransfer;
    m_queue.submit(vk::SubmitInfo{.waitSemaphoreCount = 1,
                                  .pWaitSemaphores = &*acq,
                                  .pWaitDstStageMask = &waitStage,
                                  .commandBufferCount = 1,
                                  .pCommandBuffers = &*m_cmd,
                                  .signalSemaphoreCount = 1,
                                  .pSignalSemaphores = &*rd},
                   *m_fence);

    const auto pres = m_queue.presentKHR(
        vk::PresentInfoKHR{.waitSemaphoreCount = 1,
                           .pWaitSemaphores = &*rd,
                           .swapchainCount = 1,
                           .pSwapchains = &*m_swapchain,
                           .pImageIndices = &imageIndex});
    if (pres == vk::Result::eErrorOutOfDateKHR)
      m_device.waitIdle();
  }

  void recordCommands(uint32_t imageIndex) {
    auto &cb = m_cmd;
    cb.reset();
    cb.begin({});

    // Make the writes of the SYCL kernel (host writes, as far as Vulkan is
    // concerned) visible to the transfer read below.
    const vk::BufferMemoryBarrier2 bufBar{
        .srcStageMask = vk::PipelineStageFlagBits2::eHost,
        .srcAccessMask = vk::AccessFlagBits2::eHostWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = *m_sharedBuf,
        .offset = 0,
        .size = m_sharedBytes};
    const vk::ImageMemoryBarrier2 toTransferDst =
        imgBar(*m_interImg, vk::ImageLayout::eUndefined,
               vk::ImageLayout::eTransferDstOptimal,
               vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
               vk::PipelineStageFlagBits2::eTransfer,
               vk::AccessFlagBits2::eTransferWrite);
    cb.pipelineBarrier2(
        vk::DependencyInfo{.bufferMemoryBarrierCount = 1,
                           .pBufferMemoryBarriers = &bufBar,
                           .imageMemoryBarrierCount = 1,
                           .pImageMemoryBarriers = &toTransferDst});

    const vk::BufferImageCopy2 copy{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {m_extent.width, m_extent.height, 1}};
    cb.copyBufferToImage2(vk::CopyBufferToImageInfo2{
        .srcBuffer = *m_sharedBuf,
        .dstImage = *m_interImg,
        .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
        .regionCount = 1,
        .pRegions = &copy});

    const std::array<vk::ImageMemoryBarrier2, 2> pre{
        imgBar(*m_interImg, vk::ImageLayout::eTransferDstOptimal,
               vk::ImageLayout::eTransferSrcOptimal,
               vk::PipelineStageFlagBits2::eTransfer,
               vk::AccessFlagBits2::eTransferWrite,
               vk::PipelineStageFlagBits2::eTransfer,
               vk::AccessFlagBits2::eTransferRead),
        imgBar(m_swapImages[imageIndex], vk::ImageLayout::eUndefined,
               vk::ImageLayout::eTransferDstOptimal,
               vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
               vk::PipelineStageFlagBits2::eTransfer,
               vk::AccessFlagBits2::eTransferWrite)};
    cb.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 2,
                                           .pImageMemoryBarriers = pre.data()});

    const vk::ImageSubresourceLayers sub{vk::ImageAspectFlagBits::eColor, 0, 0,
                                         1};
    const vk::Offset3D ext{int32_t(m_extent.width), int32_t(m_extent.height),
                           1};
    const vk::ImageBlit2 region{
        .srcSubresource = sub,
        .srcOffsets = std::array<vk::Offset3D, 2>{vk::Offset3D{0, 0, 0}, ext},
        .dstSubresource = sub,
        .dstOffsets = std::array<vk::Offset3D, 2>{vk::Offset3D{0, 0, 0}, ext}};
    cb.blitImage2(vk::BlitImageInfo2{
        .srcImage = *m_interImg,
        .srcImageLayout = vk::ImageLayout::eTransferSrcOptimal,
        .dstImage = m_swapImages[imageIndex],
        .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
        .regionCount = 1,
        .pRegions = &region,
        .filter = vk::Filter::eNearest});

    const vk::ImageMemoryBarrier2 toPresent = imgBar(
        m_swapImages[imageIndex], vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::ePresentSrcKHR, vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eBottomOfPipe, vk::AccessFlagBits2::eNone);
    cb.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                           .pImageMemoryBarriers = &toPresent});
    cb.end();
  }

  static vk::ImageMemoryBarrier2
  imgBar(vk::Image img, vk::ImageLayout o, vk::ImageLayout n,
         vk::PipelineStageFlags2 ss, vk::AccessFlags2 sa,
         vk::PipelineStageFlags2 ds, vk::AccessFlags2 da) {
    return vk::ImageMemoryBarrier2{
        .srcStageMask = ss,
        .srcAccessMask = sa,
        .dstStageMask = ds,
        .dstAccessMask = da,
        .oldLayout = o,
        .newLayout = n,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
  }

  void cleanup() {
    m_device.waitIdle();
    m_sycl.reset(); // must die before the Vulkan memory it writes into
    m_renderDone.clear();
    m_imageAvail.clear();
    m_fence = nullptr;
    m_cmd = nullptr;
    m_sharedBuf = nullptr;
    m_sharedMem = nullptr;
    m_sharedMapped = nullptr;
    m_interImg = nullptr;
    m_interMem = nullptr;
    m_cmdPool = nullptr;
    m_swapchain = nullptr;
    m_queue = nullptr;
    m_device = nullptr;
    m_surface = nullptr;
    m_debug = nullptr;
    m_instance = nullptr;
    if (m_window)
      glfwDestroyWindow(m_window);
    glfwTerminate();
  }

  uint32_t findMemoryType(uint32_t bits, vk::MemoryPropertyFlags props) const {
    const auto mp = m_phys.getMemoryProperties();
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
      if ((bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & props) == props)
        return i;
    throw std::runtime_error("no suitable memory type");
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT sev,
                VkDebugUtilsMessageTypeFlagsEXT,
                const VkDebugUtilsMessengerCallbackDataEXT *d, void *) {
    if (sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
      std::cerr << "[VK] " << d->pMessage << '\n';
    return VK_FALSE;
  }
};

// ---------------------------------------------------------------------------
static void listDevices() {
  for (const auto &plat : sycl::platform::get_platforms()) {
    std::cout << "platform: " << plat.get_info<sycl::info::platform::name>()
              << '\n';
    for (const auto &dev : plat.get_devices())
      std::cout << "  device: " << dev.get_info<sycl::info::device::name>()
                << " (system USM: "
                << (dev.has(sycl::aspect::usm_system_allocations) ? "yes" : "no")
                << ")\n";
  }
}

int main(int argc, char **argv) {
  std::string device = "default";
  std::string interop;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("--device=", 0) == 0) {
      device = a.substr(9);
    } else if (a.rfind("--interop=", 0) == 0) {
      interop = a.substr(10);
    } else if (a == "--list-devices") {
      listDevices();
      return EXIT_SUCCESS;
    } else {
      std::cerr << "usage: " << argv[0]
                << " [--device=cpu|gpu|default] [--interop=zero-copy|copy] "
                   "[--list-devices]\n";
      return EXIT_FAILURE;
    }
  }
  if (!interop.empty() && interop != "zero-copy" && interop != "copy") {
    std::cerr << "unknown --interop=" << interop
              << " (expected zero-copy or copy)\n";
    return EXIT_FAILURE;
  }

  std::cout << "=====================================================\n"
               " Vulkan rendering powered by SYCL (AdaptiveCpp)\n"
               "=====================================================\n";
  try {
    ForestApp app(device, interop);
    app.run();
  } catch (const std::exception &e) {
    std::cerr << "Fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
