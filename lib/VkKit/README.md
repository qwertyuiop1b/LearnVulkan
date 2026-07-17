# VkKit

`VkKit` 是一个面向 Vulkan 1.3 的 C++17 图形基础库。它位于 `vulkan_graphics` 命名空间，围绕以下目标设计：

- 用 RAII 管理 Vulkan/VMA 资源，避免应用层逐个调用 `vkDestroy*`。
- 用小而明确的对象封装资源所有权，不隐藏 Vulkan command buffer 与 native handle。
- 以 Dynamic Rendering 为默认渲染模型，不要求应用创建 `VkRenderPass` 或 framebuffer。
- 将 GPU 资源、描述符、管线、交换链和逐帧同步分层，使其可被其他 Vulkan 应用复用。

> `VkKit` 的对象不是线程安全的，除 `UploadContext` 会串行化提交外，其余对象应由调用方在外部同步。所有依赖 `VulkanContext` 的对象都必须先于 `VulkanContext` 销毁。

## 目录

- [依赖与构建](#依赖与构建)
- [架构](#架构)
- [资源所有权与生命周期](#资源所有权与生命周期)
- [公开类型参考](#公开类型参考)
- [典型用法](#典型用法)
- [逐帧渲染流程](#逐帧渲染流程)
- [当前边界](#当前边界)

## 依赖与构建

### 依赖

| 依赖 | 用途 |
| --- | --- |
| Vulkan SDK / loader | Vulkan 头文件、loader 与设备 API |
| GLFW 3.3+ | 创建 window surface |
| Vulkan Memory Allocator (VMA) | `Buffer` 和 `Image` 的内存分配 |

VMA 以头文件方式使用。作为本仓库的子目录构建时，CMake 默认使用
`external/VulkanMemoryAllocator/include/vk_mem_alloc.h`；请先初始化 submodule：

```bash
git submodule update --init --recursive
```

独立使用 VkKit 时，CMake 会在 Vulkan include 路径及其 `vma/` 子目录中查找
`vk_mem_alloc.h`，也可以显式指定其所在目录：

```bash
cmake -S lib/VkKit -B build/vkkit \
  -DVKKIT_VMA_INCLUDE_DIR=/path/to/VulkanMemoryAllocator/include
```

### 作为子目录使用

```cmake
add_subdirectory(lib/VkKit)

target_link_libraries(MyApplication PRIVATE VkKit::VkKit)
```

### 独立构建与安装

```bash
cmake -S lib/VkKit -B build/vkkit -DVKKIT_INSTALL=ON
cmake --build build/vkkit
cmake --install build/vkkit --prefix /your/prefix
```

安装后，消费者可以使用：

```cmake
find_package(VkKit CONFIG REQUIRED)
target_link_libraries(MyApplication PRIVATE VkKit::VkKit)
```

`VkKit` 的 CMake package 会继续查找 Vulkan、GLFW 和 VMA。

## 架构

```text
GLFWwindow
    │
    ▼
VulkanContext ──────────────► graphics / present / compute Queue
    │
    ├──► VulkanAllocator ────► Buffer / Image ────► Texture (Image + Sampler)
    │             │
    │             └─────────► VMA allocation
    │
    ├──► ImageStateTracker ─► 每个 mip/layer 的 layout、stage、access 状态
    │
    ├──► CommandPool ───────► UploadContext
    │
    ├──► ShaderModule + DescriptorSetLayout ───► PipelineLayout ───► GraphicsPipeline
    │                         │
    │                         └───────────────► DescriptorPool ───► DescriptorSet
    │
    └──► Swapchain ─────────► RenderTarget (每张 swapchain image 的非拥有视图)
                                  │
                                  ▼
                         FrameScheduler ───► FrameContext × N
                         acquire → record → Dynamic Rendering → submit → present
```

### 分层原则

1. **Context 层**：创建 instance、surface、physical device、logical device 和 queue。
2. **内存与资源层**：`VulkanAllocator` 统一接入 VMA；`Buffer` 与 `Image` 只管理自己的 Vulkan resource + allocation。
3. **上传层**：`UploadContext` 用同步的一次性命令提交完成 staging copy 与有限的 image layout transition。
4. **着色与绑定层**：shader、descriptor、pipeline layout 和 graphics pipeline 可以独立组合。
5. **呈现与帧层**：`Swapchain` 管理可呈现 image/view；`FrameScheduler` 管理多帧并行、同步和 Dynamic Rendering 的基础流程。

## 资源所有权与生命周期

### RAII 规则

- 除 `RenderTarget` 外，公开资源对象都拥有底层 Vulkan/VMA resource；析构函数会自动释放。
- 大多数资源类不可复制、可移动。移动后源对象不再有效。
- `DescriptorPool` 与已分配的 `DescriptorSet` 共享内部 pool state：即使 pool 包装对象提前移动或销毁，set 仍会在自己析构时正确释放；但它们都必须早于 `VulkanContext` 销毁。
- `RenderTarget` 不拥有 `VkImage` 或 `VkImageView`，其有效期仅到所属 `Swapchain` 被销毁或重建为止。

### 推荐销毁顺序

```text
FrameScheduler
GraphicsPipeline / PipelineLayout / DescriptorSet / DescriptorPool
Swapchain
Texture / Image / Buffer / Sampler / UploadContext / CommandPool
VulkanAllocator
VulkanContext
GLFWwindow
```

应用退出或销毁整个资源图前，应先确保 GPU 不再使用资源，例如调用 `context.waitIdle()`。

### Native handle

各类 `handle()` 或 `nativeHandle()` 返回的是**借用** handle，不能由调用方销毁。它们用于调用当前未封装的 Vulkan API，或与现有 Vulkan 代码集成。

## 公开类型参考

### 1. Context、队列与分配器

| 类型 | 作用 | 关键点 |
| --- | --- | --- |
| `VulkanContextCreateInfo` | 创建 context 的参数 | window、API version、validation、dynamic rendering、sampler anisotropy、device extensions |
| `Queue` | 逻辑设备 queue 的描述 | `handle`、`familyIndex`、timestamp 支持信息 |
| `VulkanContext` | instance/surface/physical device/device/queues 的根所有者 | 默认请求 Vulkan 1.3、swapchain extension、Dynamic Rendering 与 Synchronization2 |
| `VulkanAllocator` | VMA allocator 的 RAII 包装 | 为 buffer/image 分配内存；可查询统计与预算 |
| `HeapStatistics` / `AllocatorStatistics` / `MemoryBudget` | VMA 内存可观测性数据 | 用于调试内存压力和预算 |

`VulkanContext` 在创建时选择同时满足 surface present、device extension、swapchain 支持和所请求 feature 的物理设备。默认的 `requireDynamicRendering = true` 与 `requireSynchronization2 = true` 会要求设备支持 Dynamic Rendering 和 Synchronization2。

### 2. Buffer、Image、Sampler 与 Texture

| 类型 | 作用 | 关键点 |
| --- | --- | --- |
| `BufferCreateInfo` | buffer 创建参数 | size、usage、内存用途、host access、persistent map、sharing mode |
| `Buffer` | VMA 管理的 `VkBuffer` | `write`、`flush`、`invalidate`；支持 GPU-only、CPU-to-GPU、GPU-to-CPU |
| `ImageCreateInfo` | 2D image 创建参数 | extent、format、usage、aspect、mip/layer、samples、tiling、queue family |
| `Image` | VMA 管理的 `VkImage` 与默认 image view | 记录当前 layout，供 `UploadContext` 使用 |
| `ImageSubresourceState` | 一个 image subresource 的同步状态 | layout、Synchronization2 stage/access mask、queue family |
| `ImageStateTracker` | 按 mip/layer 跟踪 image 状态 | 为后续 subresource barrier 与多 pass 渲染提供基础 |
| `SamplerCreateInfo` | sampler 过滤、寻址、LOD、比较、各向异性参数 | 启用 anisotropy 前 context 必须实际启用该 feature |
| `Sampler` | `VkSampler` RAII 包装 | 可与 `Image` 组合使用 |
| `TextureCreateInfo` | 内存中的纹理像素参数 | extent、像素数据、字节数、format、sampler 参数 |
| `Texture` | `Image + Sampler` 的组合资源 | 内部创建 staging buffer，上传后转为 shader-read layout |

#### Buffer 内存策略

- `BufferMemoryUsage::GpuOnly`：显存优先。通常使用 staging buffer + `UploadContext::copyBuffer` 上传。
- `BufferMemoryUsage::CpuToGpu`：CPU 可写。适合 uniform、动态 vertex data、staging buffer。
- `BufferMemoryUsage::GpuToCpu`：CPU 可读。适合 readback。
- `persistentMap = true`：创建后长期保持映射，可通过 `mappedData()` 获取指针；非 coherent memory 写入后仍需要 `flush()`。

#### Image layout 约束

`Image` 内部通过 `ImageStateTracker` 按 mip/layer 记录 layout、stage 与 access；`UploadContext` 通过 Synchronization2 barrier 更新这些状态。当前 `UploadContext::transitionImageLayout` 仍只提供以下同步上传路径：

```text
Undefined → TransferDstOptimal → ShaderReadOnlyOptimal
```

`copyBufferToImage` 当前用于单 mip image。更复杂的 mip generation、storage image、depth/stencil transition 或异步 transfer queue 提交，应在后续扩展专用 command API。

### 3. CommandPool 与 UploadContext

| 类型 | 作用 | 关键点 |
| --- | --- | --- |
| `CommandPool` | 指定 queue family 的 `VkCommandPool` RAII 包装 | 可用于自定义 command buffer 分配 |
| `UploadContext` | 同步上传执行器 | 内部有 transient pool、fence 和 mutex；复制完成后才返回 |

`UploadContext` 适合初始化阶段或低频资源上传。它提供：

- `copyBuffer(source, destination, ...)`
- `transitionImageLayout(image, oldLayout, newLayout)`
- `copyBufferToImage(source, destination)`

因为它每次上传都会提交并等待，不应当用作每帧的大量 streaming 上传器。

### 4. Shader、描述符与管线布局

| 类型 | 作用 | 关键点 |
| --- | --- | --- |
| `ShaderModule` | `VkShaderModule` RAII 包装 | 从 SPIR-V word vector 或 `.spv` 文件创建；验证 magic 与字节对齐 |
| `DescriptorBinding` | 单个 descriptor binding 描述 | binding、type、array count、shader stages |
| `DescriptorSetLayout` | `VkDescriptorSetLayout` 包装 | 创建时检查 binding 唯一性、数量和 stage flags |
| `DescriptorPoolSize` / `DescriptorPoolCreateInfo` | descriptor pool 容量参数 | 以 descriptor type 和 max set 数量定义容量 |
| `DescriptorPool` | descriptor pool 的共享所有者 | `allocate(layout)` 返回 RAII `DescriptorSet` |
| `DescriptorSet` | 已分配的 descriptor set | 可写 buffer 或 combined image sampler descriptor |
| `PushConstantRange` | push constant 的 stage/offset/size | offset 与 size 必须 4-byte 对齐 |
| `PipelineLayout` | set layout + push constant range 的组合 | 创建 `VkPipelineLayout`，验证设备限制与 range overlap |

`DescriptorSet::writeBuffer` 适合 uniform/storage buffer（含 dynamic 变体）。`writeTexture` 当前支持 `CombinedImageSampler`，要求纹理 image 已位于 `ShaderReadOnlyOptimal`。

### 5. GraphicsPipeline

| 类型 | 作用 | 关键点 |
| --- | --- | --- |
| `GraphicsShaderStage` | shader module、stage、entry point | 当前仅允许 vertex 与 fragment stage |
| `VertexBindingDescription` / `VertexAttributeDescription` | 顶点输入描述 | binding、stride、input rate、location、format、offset |
| `ColorBlendAttachmentState` | 单个 color attachment 的 blend 状态 | 包含 color/alpha factor、op、write mask |
| `StencilFaceState` | front/back stencil 状态 | fail/pass/depth-fail、compare、mask/reference |
| `GraphicsPipelineCreateInfo` | 完整 graphics pipeline 状态 | shader、layout、vertex input、raster、depth/stencil、blend、attachment formats |
| `GraphicsPipeline` | `VkPipeline` RAII 包装 | 使用 Dynamic Rendering 创建，不绑定 render pass |

创建 pipeline 时，`colorAttachmentFormats` 必须与实际 Dynamic Rendering color attachment format 对应。渲染 swapchain 时通常设置为：

```cpp
pipelineInfo.colorAttachmentFormats = {swapchain.format()};
```

viewport 与 scissor 被声明为动态状态，应用必须在 command buffer 中调用 `vkCmdSetViewport` 和 `vkCmdSetScissor`。当前实现为保证 feature 明确性，仅接受 `PolygonMode::eFill` 与 `lineWidth == 1.0f`。

### 6. Swapchain 与 RenderTarget

| 类型 | 作用 | 关键点 |
| --- | --- | --- |
| `SwapchainCreateInfo` | surface format、present mode、image count、usage、desired extent | 默认偏好 mailbox，再回退到 FIFO |
| `SwapchainStatus` | acquire/present 结果 | `eSuccess`、`eSuboptimal`、`eOutOfDate` |
| `SwapchainAcquireResult` | acquire 状态和 image index | `eOutOfDate` 时不应使用 image index |
| `RenderTarget` | 单张 swapchain image/view 的借用视图 | 不拥有资源；重建 swapchain 后旧引用立即失效 |
| `Swapchain` | `VkSwapchainKHR` 和所有 swapchain image view 的所有者 | 支持 acquire、present、RAII 重建 |

`Swapchain::recreate()` 内部等待 device idle，并采用 `oldSwapchain` 创建新资源成功后再释放旧资源。若已使用 `FrameScheduler`，必须调用 `FrameScheduler::recreateSwapchain()`，而不是直接调用 `Swapchain::recreate()`，以同步刷新 scheduler 的 image/fence/layout 缓存。

### 7. FrameContext 与 FrameScheduler

| 类型 | 作用 | 关键点 |
| --- | --- | --- |
| `FrameContext` | 一帧独占的命令与同步资源 | 一个 resettable command pool、primary command buffer、两个 binary semaphore、一个 fence |
| `FrameSchedulerCreateInfo` | scheduler 参数 | swapchain 指针和 frames-in-flight 数量（默认 2） |
| `FrameBeginResult` | `beginFrame()` 的结果 | status、image index、当前 frame、当前 render target |
| `DynamicRenderingInfo` | color/depth/stencil attachment 的 load/store/clear 参数 | color attachment 固定为当前 swapchain render target |
| `FrameScheduler` | 全帧生命周期管理器 | acquire、per-image fence、layout barrier、submit、present、swapchain recreate |

每个 `FrameContext` 的 semaphore 职责固定：

```text
imageAvailableSemaphore: vkAcquireNextImageKHR → graphics queue submit
renderFinishedSemaphore: graphics queue submit → vkQueuePresentKHR
inFlightFence:           graphics queue submit → CPU 等待后复用本帧 command buffer
```

`FrameScheduler` 额外维护 `imageInFlightFences_` 与每张 swapchain image 的 `ImageStateTracker` 状态，所以当 swapchain image 数多于或少于 frames-in-flight 数时，仍不会把同一 image 同时交给多个 GPU submission 写入。提交与 layout barrier 使用 `VkSubmitInfo2`、`VkImageMemoryBarrier2` 和 `vkCmdPipelineBarrier2`。

## 典型用法

以下示例展示最小的初始化、pipeline 创建和一帧绘制结构。错误处理策略由应用决定，这里用异常直接向上传播。

```cpp
#include <graphics/frame_scheduler.hpp>
#include <graphics/graphics_pipeline.hpp>
#include <graphics/shader_module.hpp>
#include <graphics/swapchain.hpp>
#include <graphics/vulkan_allocator.hpp>
#include <graphics/vulkan_context.hpp>

using namespace vulkan_graphics;

// GLFW 已初始化，window 已用 GLFW_NO_API 创建。
VulkanContextCreateInfo contextInfo{};
contextInfo.window = window;
contextInfo.applicationName = "My Vulkan App";
VulkanContext context{contextInfo};

VulkanAllocator allocator{context};

SwapchainCreateInfo swapchainInfo{};
swapchainInfo.desiredExtent = {windowWidth, windowHeight};
Swapchain swapchain{context, swapchainInfo};

FrameSchedulerCreateInfo schedulerInfo{};
schedulerInfo.swapchain = &swapchain;
schedulerInfo.framesInFlight = 2;
FrameScheduler frameScheduler{context, schedulerInfo};

ShaderModule vertexShader = ShaderModule::fromFile(context, "shaders/triangle.vert.spv");
ShaderModule fragmentShader = ShaderModule::fromFile(context, "shaders/triangle.frag.spv");

PipelineLayoutCreateInfo pipelineLayoutInfo{};
PipelineLayout pipelineLayout{context, pipelineLayoutInfo};

GraphicsPipelineCreateInfo pipelineInfo{};
pipelineInfo.layout = &pipelineLayout;
pipelineInfo.shaderStages = {
    {&vertexShader, vk::ShaderStageFlagBits::eVertex, "main"},
    {&fragmentShader, vk::ShaderStageFlagBits::eFragment, "main"},
};
pipelineInfo.colorAttachmentFormats = {swapchain.format()};
GraphicsPipeline pipeline{context, pipelineInfo};
```

## 逐帧渲染流程

```cpp
void drawFrame(FrameScheduler& scheduler,
               GraphicsPipeline& pipeline,
               VkExtent2D framebufferExtent) {
    const FrameBeginResult beginResult = scheduler.beginFrame();
    if (beginResult.status == SwapchainStatus::eOutOfDate) {
        scheduler.recreateSwapchain(framebufferExtent);
        return;
    }

    VkCommandBuffer commandBuffer = beginResult.frame->commandBuffer();

    DynamicRenderingInfo renderingInfo{};
    renderingInfo.colorClearValue = {{0.03f, 0.05f, 0.10f, 1.0f}};
    scheduler.beginDynamicRendering(renderingInfo);

    VkViewport viewport{};
    viewport.width = static_cast<float>(beginResult.renderTarget->extent().width);
    viewport.height = static_cast<float>(beginResult.renderTarget->extent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, beginResult.renderTarget->extent()};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.nativeHandle());

    // vkCmdBindDescriptorSets(...)
    // vkCmdBindVertexBuffers(...)
    // vkCmdDraw(...)

    scheduler.endDynamicRendering();
    const SwapchainStatus presentStatus = scheduler.endFrame();
    if (presentStatus != SwapchainStatus::eSuccess)
        scheduler.recreateSwapchain(framebufferExtent);
}
```

调用约束：

1. 每次 `beginFrame()` 成功后，必须恰好调用一次 `endFrame()`。
2. `beginDynamicRendering()` 与 `endDynamicRendering()` 必须成对调用，并且在 `endFrame()` 前结束。
3. 当前 `FrameScheduler` 管理 swapchain color image 的布局：`Undefined/PresentSrcKHR → ColorAttachmentOptimal → PresentSrcKHR`。不要在同一 image 上绕过 scheduler 执行冲突的 layout transition。
4. 收到 `eOutOfDate` 时不要提交当前帧；重建 swapchain 后从下一帧重新 acquire。
5. 收到 `eSuboptimal` 时当前帧仍可正常提交和 present，但应尽快在合适的窗口尺寸下调用 `recreateSwapchain()`。

## 当前边界

`VkKit` 已覆盖基础 Vulkan 资源与帧循环，但以下能力尚未封装：

- GLFW window 生命周期、事件循环和 framebuffer-size 回调。
- SPIR-V 编译、shader reflection、自动生成 descriptor/pipeline layout。
- descriptor indexing、bindless descriptor、push descriptor。
- graphics pipeline cache、pipeline library、compute pipeline、ray tracing pipeline。
- 自动创建 depth/MSAA render target；`FrameScheduler` 可接受外部 depth/stencil view，但其 image 创建和 layout transition 由应用负责。
- timeline semaphore、异步 transfer queue、批量/异步 resource upload。
- render graph、资源别名、barrier 自动推导、GPU profiler。

这些边界是有意保留的：当前库优先提供清晰的资源边界和可组合的 Vulkan 基元，而不是在基础层隐藏全部 Vulkan 细节。
