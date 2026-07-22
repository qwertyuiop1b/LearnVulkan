# VkKit 信号量重用问题分析

## 问题描述

运行 `vkit_triangle` 时出现 Vulkan 验证层错误：

```
[VK ERROR] vkQueueSubmit2(): pSubmits[0].pSignalSemaphoreInfos[0].semaphore is being signaled
by VkQueue, but it may still be in use by VkSwapchainKHR.
```

## 根本原因

这是 **VkKit 库的已知架构问题**，不是示例代码的问题。

### 问题场景

当交换链图像数量 > 帧并行数时会触发：

```
配置：
- 交换链图像数：3 张（image 0, 1, 2）
- Frames in flight：2 个（frame A, B）
- 每个 frame 有：imageAvailableSemaphore, renderFinishedSemaphore

时间线：
Frame A: acquire(img0) → render → submit(semA) → present(semA_render)
Frame B: acquire(img1) → render → submit(semB) → present(semB_render)
Frame A: acquire(img2) → render → submit(semA) ← 问题！
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         semA 可能仍被 img0 的 present 操作使用
```

### VkKit 的信号量架构

查看源代码 `frame_scheduler.cpp`：

```cpp
// 每个 FrameContext 有固定的信号量
class FrameContext {
    VkSemaphore imageAvailableSemaphore_;   // 固定绑定到此帧
    VkSemaphore renderFinishedSemaphore_;   // 固定绑定到此帧
    VkFence inFlightFence_;
};

// FrameScheduler 使用循环的帧
std::vector<std::unique_ptr<FrameContext>> frames_;  // 2 个帧
currentFrameIndex_ = (currentFrameIndex_ + 1) % frames_.size();
```

问题：
- **2 个帧 → 2 组信号量**
- **3 张图像 → 需要 3 组信号量**（或使用 fence 同步）

### Vulkan 规范要求

根据 [Vulkan 信号量重用指南](https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html)：

> Binary semaphores must be unsignaled when submitted again.

当 `present(imageIndex=0, semaphore_A)` 后，在 `imageIndex=0` 再次被 `acquire` 之前，
`semaphore_A` 可能仍在使用中，不能重用于其他图像。

## 标准解决方案

### 方案 1：每张图像一个信号量（推荐）

```cpp
// 正确的架构
struct PerImageSemaphores {
    VkSemaphore imageAvailable;
    VkSemaphore renderFinished;
};
std::vector<PerImageSemaphores> imageSeamphores_;  // 与交换链图像数量一致

// 使用时根据 imageIndex 索引
uint32_t imageIndex = acquireNextImage();
VkSemaphore imageAvailable = imageSemaphores_[imageIndex].imageAvailable;
```

### 方案 2：使用 VK_KHR_swapchain_maintenance1 扩展

此扩展允许在 present 时使用 fence，而不是依赖信号量。

### 方案 3：确保帧数 >= 图像数

```cpp
FrameSchedulerCreateInfo schedulerInfo{};
schedulerInfo.framesInFlight = swapchain.imageCount();  // 3 帧对 3 图像
```

缺点：增加延迟和内存开销。

## VkKit 当前实现

VkKit 使用了 **fence 跟踪**来部分缓解问题：

```cpp
// frame_scheduler.cpp:43-47
const VkFence imageFence = imageInFlightFences_[acquisition.imageIndex];
if (imageFence != VK_NULL_HANDLE && imageFence != frame.inFlightFence()) {
    vkWaitForFences(..., &imageFence, ...);  // 等待之前使用此图像的帧完成
}
```

这确保了：
- **图像**不会被同时写入
- **Fence** 同步正确

但是没有解决：
- **信号量**仍可能被重用（present 操作是异步的）

## 为什么程序仍能运行？

虽然违反了 Vulkan 规范，但在实践中：
1. Fence 等待确保了图像不会同时被写入
2. GPU 通常足够快，信号量实际上已经完成
3. 只有验证层会报错，release 模式可能不会崩溃

但这是**未定义行为**，在某些驱动/硬件上可能导致问题。

## 对示例代码的影响

### 你的示例代码是正确的 ✅

```cpp
// vkit_triangle.cpp
FrameSchedulerCreateInfo schedulerInfo{};
schedulerInfo.swapchain = swapchain_.get();
schedulerInfo.framesInFlight = 2;  // 标准配置
frameScheduler_ = std::make_unique<FrameScheduler>(*context_, schedulerInfo);
```

这是标准用法，问题在于 VkKit 库的底层实现。

## 解决建议

### 短期：禁用验证层（不推荐）

```cpp
contextInfo.enableValidation = false;  // 隐藏错误，不解决问题
```

### 中期：等待 VkKit 修复

这需要 VkKit 库的作者重构 FrameScheduler 的信号量管理。

### 长期：贡献修复到 VkKit

修改 `FrameScheduler` 实现方案 1：

```cpp
// 建议的修改（伪代码）
class FrameScheduler {
    // 改为：每个图像一组信号量
    std::vector<VkSemaphore> imageAvailableSemaphores_;  // imageCount 个
    std::vector<VkSemaphore> renderFinishedSemaphores_;  // imageCount 个
    
    // 帧只需要 fence 和 command buffer
    struct PerFrameData {
        VkFence fence;
        VkCommandBuffer commandBuffer;
    };
    std::vector<PerFrameData> frames_;  // framesInFlight 个
};
```

## 结论

✅ **这是 VkKit 库的问题**  
✅ **你的示例代码没有错**  
⚠️ **验证层错误是正确的警告**  
🔧 **需要修改 VkKit 库才能彻底解决**

## 参考资源

- [Vulkan 信号量重用指南](https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html)
- [Vulkan Spec VUID-vkQueueSubmit2-semaphore-03868](https://vulkan.lunarg.com/doc/view/latest/linux/1.3-extensions/vkspec.html#VUID-vkQueueSubmit2-semaphore-03868)
- VkKit 源码：`lib/VkKit/src/graphics/render/frame_scheduler.cpp`
