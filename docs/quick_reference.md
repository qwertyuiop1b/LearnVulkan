# 🚀 Vulkan 三角形渲染 - 快速参考

## 📊 对象依赖链表

```
创建顺序必须遵守：

Instance 
  ↓ (需要)
PhysicalDevice ← Surface (也需要)
  ↓ (需要)
Device
  ├──→ CommandPool → CommandBuffer
  ├──→ RenderPass → Pipeline
  └──→ Semaphore + Fence

Surface + PhysicalDevice + Device
  ↓ (需要)
Swapchain
  ↓ (需要)
ImageViews
  ↓ (需要)
Framebuffers (也需要 RenderPass)
```

---

## 🎨 每帧渲染的 6 步

```cpp
void drawFrame() {
    // 1️⃣ 等待前一帧完成 (GPU→CPU)
    vkWaitForFences(device, 1, &fence[frame], VK_TRUE, UINT64_MAX);
    
    // 2️⃣ 从交换链获取图像
    uint32_t imageIdx;
    vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, 
                          semaphore[frame], VK_NULL_HANDLE, &imageIdx);
    
    // 3️⃣ 重置栅栏
    vkResetFences(device, 1, &fence[frame]);
    
    // 4️⃣ 录制命令
    vkResetCommandBuffer(cmdBuffer[frame], 0);
    recordCommands(cmdBuffer[frame], imageIdx);
    
    // 5️⃣ 提交到 GPU (GPU-GPU 同步)
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence[frame]);
    
    // 6️⃣ 显示图像
    vkQueuePresentKHR(presentQueue, &presentInfo);
    
    frame = (frame + 1) % MAX_FRAMES_IN_FLIGHT;
}
```

---

## 🔐 同步原语一览

| 名称 | 用途 | 何时 Signal | 何时 Wait |
|------|------|-----------|---------|
| **Semaphore** | GPU-GPU同步 | vkQueueSubmit 完成时 | 下个 vkQueueSubmit 前 |
| **Fence** | GPU-CPU同步 | vkQueueSubmit 完成时 | CPU 调用 vkWaitForFences |

**关键：** 第一帧的 Fence 必须初始化为 `VK_FENCE_CREATE_SIGNALED_BIT`，否则永远等不到

---

## 🏗️ 对象生命周期

### 全局对象（程序周期）
```
Instance ─→ (cleanup)─→ 销毁
Device ─→ (cleanup)─→ 销毁
RenderPass ─→ (cleanup)─→ 销毁
Pipeline ─→ (cleanup)─→ 销毁
CommandPool ─→ (cleanup)─→ 销毁
```

### 可重建对象（窗口改变时）
```
Swapchain ─→ (窗口改变)─→ destroy ─→ recreate
ImageViews ─→ (随 Swapchain)─→ destroy ─→ recreate
Framebuffers ─→ (随 Swapchain)─→ destroy ─→ recreate
```

### 循环对象（每帧使用 N 个，循环利用）
```
CommandBuffer[MAX_FRAMES_IN_FLIGHT] ─(循环)─ CommandBuffer[0]
Semaphore[MAX_FRAMES_IN_FLIGHT] ─(循环)─ Semaphore[0]
Fence[MAX_FRAMES_IN_FLIGHT] ─(循环)─ Fence[0]
```

---

## ⚡ 性能优化建议

### ✅ 好的做法

```cpp
// 1. 使用多个 CommandBuffer（每帧一个）
std::vector<VkCommandBuffer> cmdBuffers(MAX_FRAMES_IN_FLIGHT);

// 2. 提前录制命令（不是每帧都重新录制）
// 如果场景不变，重用 CommandBuffer

// 3. 批量提交（一次提交多个 CommandBuffer）
VkSubmitInfo submit{};
submit.commandBufferCount = 3;  // 一次提交 3 个

// 4. 使用 VK_KHR_dynamic_rendering（减少 RenderPass 开销）
```

### ❌ 应避免的做法

```cpp
// 1. 每帧重新创建 Semaphore/Fence
vkCreateSemaphore(...);  // ❌ 不要每帧做

// 2. 每帧等待 Fence（CPU 阻塞）
vkWaitForFences(..., UINT64_MAX);  // 如果 GPU 落后 = 死机

// 3. 交换链过期不重建
// VK_ERROR_OUT_OF_DATE_KHR 一定要处理

// 4. 不设置 Fence 初始状态
VkFenceCreateInfo fenceCI{};  // ❌ 忘记 SIGNALED_BIT
```

---

## 🐛 常见 Bug 排查

| 症状 | 原因 | 解决 |
|------|------|------|
| **程序卡死** | Fence 初始未 signal | 加 `VK_FENCE_CREATE_SIGNALED_BIT` |
| **花屏/闪烁** | 同步顺序错误 | 检查 waitStages 是否为 `COLOR_ATTACHMENT_OUTPUT` |
| **内存泄漏** | 重建时没销毁旧对象 | 先 `destroySwapchain()` 再 `recreate()` |
| **验证错误** | 缺少必需扩展 | macOS 需要 `VK_KHR_portability_subset` |
| **黑屏** | RenderPass/Pipeline 配置错 | 检查 colorAttachmentRef.layout |
| **性能掉帧** | 每帧创建对象 | 改为预分配对象，循环使用 |

---

## 📝 代码模板

### 初始化顺序检查清单

- [ ] 1. `glfwInit()` + 创建窗口
- [ ] 2. `vkCreateInstance()`
- [ ] 3. `glfwCreateWindowSurface()`
- [ ] 4. `vkEnumeratePhysicalDevices()` + 选择
- [ ] 5. `vkCreateDevice()`
- [ ] 6. `vkCreateSwapchainKHR()`
- [ ] 7. `vkGetSwapchainImagesKHR()`
- [ ] 8. `vkCreateImageView()` (对每个 image)
- [ ] 9. `vkCreateRenderPass()`
- [ ] 10. `vkCreateGraphicsPipelines()`
- [ ] 11. `vkCreateFramebuffer()` (对每个 imageView)
- [ ] 12. `vkCreateCommandPool()`
- [ ] 13. `vkAllocateCommandBuffers()`
- [ ] 14. `vkCreateSemaphore()` + `vkCreateFence()`

### 清理顺序检查清单（逆序）

- [ ] 1. `vkDeviceWaitIdle()` (等待 GPU 完成)
- [ ] 2. `vkDestroySemaphore()` (所有信号量)
- [ ] 3. `vkDestroyFence()` (所有栅栏)
- [ ] 4. `vkFreeCommandBuffers()` (不需要显式销毁)
- [ ] 5. `vkDestroyCommandPool()`
- [ ] 6. `vkDestroyFramebuffer()` (所有)
- [ ] 7. `vkDestroyPipeline()`
- [ ] 8. `vkDestroyPipelineLayout()`
- [ ] 9. `vkDestroyRenderPass()`
- [ ] 10. `vkDestroyImageView()` (所有)
- [ ] 11. `vkDestroySwapchainKHR()`
- [ ] 12. `vkDestroyDevice()`
- [ ] 13. `vkDestroySurfaceKHR()`
- [ ] 14. `vkDestroyInstance()`
- [ ] 15. `glfwDestroyWindow()` + `glfwTerminate()`

---

## 🎯 Vulkan 坐标系统

```
窗口坐标系                 NDC (Normalized Device Coords)
                        
(0, 0) ─── x ───→        (-1, -1) ─ x ─→ (1, -1)
  │                        │
  y                        y
  │                        │
  ↓                        ↓
(0, H)              (-1, 1) ─────→ (1, 1)

Vulkan 的 clip space：
- x, y: [-1, 1]（中心在原点）
- z: [0, 1]（Vulkan，不是 [-1, 1] 的 OpenGL）
- 左上角是 (0, 0)，向右 x+，向下 y+
```

---

## 💡 最佳实践总结

1. **始终检查返回值** → 用 `VK_CHECK()` 宏
2. **分层设计** → 每个类负责一种职责
3. **提前分配** → 别每帧 create/destroy
4. **正确同步** → 理解 Semaphore vs Fence
5. **处理重建** → 响应 `VK_ERROR_OUT_OF_DATE_KHR`
6. **验证层开启** → 开发时启用验证，发布时禁用
7. **阅读规范** → [Vulkan 官方文档](https://vulkan.lunarg.com/doc/view/latest/mac/tutorial/html/)

