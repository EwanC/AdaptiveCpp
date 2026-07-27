# Vulkan backend – SYCL object mapping

This page documents how SYCL objects are mapped to Vulkan objects in the
AdaptiveCpp Vulkan backend.  The diagrams use
[Mermaid](https://mermaid.js.org/) and can be rendered in any modern Markdown
viewer or the mkdocs site.

---

## 1. Platform and device hierarchy

A `sycl::platform` enumerates the physical GPUs visible to a single
`VkInstance`.  Each `sycl::device` wraps one **logical** `VkDevice` created on
top of a `VkPhysicalDevice`.  One `VkDevice` per physical GPU is created at
backend initialisation time and is shared by every SYCL object that targets
that device.

```mermaid
graph TD
    subgraph SYCL
        P[sycl::platform]
        D0[sycl::device 0]
        D1[sycl::device 1]
        P --> D0
        P --> D1
    end

    subgraph Vulkan
        I[VkInstance]
        PD0[VkPhysicalDevice 0]
        PD1[VkPhysicalDevice 1]
        LD0[VkDevice 0\nlogical device]
        LD1[VkDevice 1\nlogical device]
        I --> PD0
        I --> PD1
        PD0 --> LD0
        PD1 --> LD1
    end

    P  -- "maps to" --> I
    D0 -- "maps to" --> LD0
    D1 -- "maps to" --> LD1
```

---

## 2. Queue mapping

A `sycl::queue` bound to a particular device is backed by a **single**
`VkQueue` retrieved from the compute queue family of that device's `VkDevice`.
All `sycl::queue` objects that target the same device share this one
`VkQueue`.  A `VkCommandPool` per device is used to allocate command buffers
(see §3 below).

```mermaid
graph TD
    subgraph SYCL
        Q0[sycl::queue A\ntargets device 0]
        Q1[sycl::queue B\ntargets device 0]
        Q2[sycl::queue C\ntargets device 1]
    end

    subgraph Vulkan device 0
        VQ0[VkQueue\ncompute queue family\nqueue index 0]
        CP0[VkCommandPool\nfor device 0]
        VQ0 --- CP0
    end

    subgraph Vulkan device 1
        VQ1[VkQueue\ncompute queue family\nqueue index 0]
        CP1[VkCommandPool\nfor device 1]
        VQ1 --- CP1
    end

    Q0 -- "submits via" --> VQ0
    Q1 -- "submits via" --> VQ0
    Q2 -- "submits via" --> VQ1
```

---

## 3. Command submission and synchronisation

Each `queue::submit()` call (a *command group*) is translated into a
`VkCommandBuffer` containing a **single recorded command** (kernel dispatch,
memory copy, etc.).  The command buffer is submitted to the shared `VkQueue`
via `vkQueueSubmit2`.

Synchronisation between commands (SYCL events / inter-queue dependencies) is
handled exclusively with **timeline semaphores** (`VkSemaphoreTypeTimeline`).
Every submission signals a monotonically increasing counter value on a
per-device timeline semaphore, and any dependent submission waits on that
counter value before starting execution.

```mermaid
sequenceDiagram
    participant App as SYCL application
    participant RT  as AdaptiveCpp runtime
    participant CB  as VkCommandBuffer
    participant Q   as VkQueue (shared)
    participant TS  as VkSemaphore (timeline)

    App->>RT: queue.submit(kernel A)
    RT->>CB: vkBeginCommandBuffer
    RT->>CB: vkCmdDispatch (kernel A)
    RT->>CB: vkEndCommandBuffer
    RT->>Q: vkQueueSubmit2\n  signal: timeline counter N+1
    Q-->>TS: signals counter N+1

    App->>RT: queue.submit(kernel B)\n  depends on kernel A event
    RT->>CB: vkBeginCommandBuffer
    RT->>CB: vkCmdDispatch (kernel B)
    RT->>CB: vkEndCommandBuffer
    RT->>Q: vkQueueSubmit2\n  wait: timeline counter N+1\n  signal: timeline counter N+2
    TS-->>Q: unblocks when counter ≥ N+1
    Q-->>TS: signals counter N+2
```

---

## 4. Full object map summary

The table below consolidates all mappings described above.

| SYCL object | Vulkan object(s) | Notes |
|---|---|---|
| `sycl::platform` | `VkInstance` | One instance per backend plugin load |
| `sycl::device` | `VkPhysicalDevice` + `VkDevice` | One logical device per physical GPU |
| `sycl::context` | `VkDevice` (shared) | Context maps to the same `VkDevice` as its devices |
| `sycl::queue` | `VkQueue` (shared) + `VkCommandPool` | All queues targeting the same device share one `VkQueue` |
| `queue::submit()` | `VkCommandBuffer` (one command) | Allocated from the per-device `VkCommandPool` |
| `sycl::event` | `VkSemaphore` (timeline) counter value | Monotonically increasing counter per device |
| USM allocation | `VkDeviceMemory` / `VmaAllocation` | Device or host memory via Vulkan Memory Allocator |

```mermaid
graph LR
    subgraph SYCL objects
        SP[sycl::platform]
        SD[sycl::device]
        SC[sycl::context]
        SQ[sycl::queue]
        SS[queue::submit]
        SE[sycl::event]
        USM[USM allocation]
    end

    subgraph Vulkan objects
        VI[VkInstance]
        VPD[VkPhysicalDevice]
        VLD[VkDevice\nlogical device]
        VQ[VkQueue\nshared per device]
        VCB[VkCommandBuffer\none command]
        VTS[VkSemaphore\ntimeline]
        VMem[VkDeviceMemory]
    end

    SP  --> VI
    SD  --> VPD
    SD  --> VLD
    SC  --> VLD
    SQ  --> VQ
    SS  --> VCB
    SE  --> VTS
    USM --> VMem
```
