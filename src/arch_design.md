# Vulkan 三角形渲染架构设计

## 一、对象分层结构

```
┌─────────────────────────────────────────────────┐
│          Application / App (最上层)              │
│  ├─ initWindow()                                │
│  ├─ initVulkan()                                │
│  ├─ mainLoop()                                  │
│  └─ cleanup()                                   │
└─────────────────────────────────────────────────┘
          ↓ 依赖于
┌─────────────────────────────────────────────────┐
│   层 1：系统对象（Context）                      │
│   ├─ VkInstance                                 │
│   ├─ VkSurfaceKHR                              │
│   ├─ VkPhysicalDevice                          │
│   └─ VkDevice                                   │
│   💡 职责：初始化一次，程序周期存活              │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│   层 2：渲染框架对象（RenderFramework）         │
│   ├─ VkRenderPass      (渲染通道定义)          │
│   ├─ VkPipeline        (图形管线)              │
│   ├─ VkPipelineLayout  (管线布局)              │
│   └─ VkCommandPool     (命令池)                │
│   💡 职责：描述"如何渲染"                       │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│   层 3：交换链对象（SwapChain）[可重建]         │
│   ├─ VkSwapchainKHR    (交换链)                │
│   ├─ VkImage[]         (图像)                  │
│   ├─ VkImageView[]     (图像视图)              │
│   └─ VkFramebuffer[]   (帧缓冲数组)            │
│   💡 职责：屏幕分辨率改变时重建                 │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│   层 4：帧资源对象（FrameResource）[循环使用]   │
│   ├─ VkCommandBuffer   (命令缓冲)              │
│   ├─ VkSemaphore       (信号量 x2)            │
│   └─ VkFence           (栅栏)                  │
│   💡 职责：每帧重复使用（N帧在飞行）           │
└─────────────────────────────────────────────────┘
```

---

## 二、对象封装建议

### 2.1 方案A：逐层封装（推荐用于中大型项目）

```cpp
// ─────────────────────────────────────────────────────────────
// 层 1：VkContext - 基础上下文
// ─────────────────────────────────────────────────────────────
class VkContext {
public:
    void init(const std::string& appName) {
        createInstance();
        createDevice();
    }
    void cleanup() {
        vkDestroyInstance(instance_, nullptr);
    }
    
    VkInstance instance() const { return instance_; }
    VkDevice device() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkQueue graphicsQueue() const { return graphicsQueue_; }
    
private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
};

// ─────────────────────────────────────────────────────────────
// 层 2：VkRenderFramework - 渲染框架
// ─────────────────────────────────────────────────────────────
class VkRenderFramework {
public:
    void init(VkContext& ctx, VkSurfaceKHR surface) {
        context_ = &ctx;
        surface_ = surface;
        
        createRenderPass();
        createGraphicsPipeline();
        createCommandPool();
    }
    void cleanup() {
        vkDestroyPipeline(context_->device(), pipeline_, nullptr);
        vkDestroyRenderPass(context_->device(), renderPass_, nullptr);
    }
    
    VkRenderPass renderPass() const { return renderPass_; }
    VkPipeline pipeline() const { return pipeline_; }
    
private:
    VkContext* context_;
    VkSurfaceKHR surface_;
    VkRenderPass renderPass_;
    VkPipeline pipeline_;
    VkPipelineLayout pipelineLayout_;
};

// ─────────────────────────────────────────────────────────────
// 层 3：VkSwapChainManager - 交换链管理器（可重建）
// ─────────────────────────────────────────────────────────────
class VkSwapChainManager {
public:
    void init(VkContext& ctx, VkRenderFramework& framework) {
        context_ = &ctx;
        framework_ = &framework;
        recreate();
    }
    void cleanup() {
        destroySwapchain();
    }
    
    void recreate() {
        // 销毁旧的
        destroySwapchain();
        
        // 创建新的
        createSwapchain();
        createImageViews();
        createFramebuffers();
    }
    
    VkSwapchainKHR swapchain() const { return swapchain_; }
    const std::vector<VkFramebuffer>& framebuffers() const { return framebuffers_; }
    VkExtent2D extent() const { return extent_; }
    
private:
    VkContext* context_;
    VkRenderFramework* framework_;
    VkSwapchainKHR swapchain_;
    std::vector<VkImage> images_;
    std::vector<VkImageView> imageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    VkExtent2D extent_;
    
    void destroySwapchain() { /* 销毁逻辑 */ }
    void createSwapchain() { /* 创建逻辑 */ }
    void createImageViews() { /* 创建逻辑 */ }
    void createFramebuffers() { /* 创建逻辑 */ }
};

// ─────────────────────────────────────────────────────────────
// 层 4：VkFrameResource - 单帧资源
// ─────────────────────────────────────────────────────────────
struct VkFrameResource {
    VkCommandBuffer commandBuffer;
    VkSemaphore imageAvailableSem;
    VkSemaphore renderFinishedSem;
    VkFence inFlightFence;
    
    void init(VkDevice device, VkCommandPool cmdPool) {
        // 分配command buffer
        VkCommandBufferAllocateInfo ai{...};
        vkAllocateCommandBuffers(device, &ai, &commandBuffer);
        
        // 创建信号量和栅栏
        VkSemaphoreCreateInfo semCI{...};
        vkCreateSemaphore(device, &semCI, nullptr, &imageAvailableSem);
        vkCreateSemaphore(device, &semCI, nullptr, &renderFinishedSem);
        
        VkFenceCreateInfo fenceCI{...};
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device, &fenceCI, nullptr, &inFlightFence);
    }
    
    void cleanup(VkDevice device) {
        vkDestroySemaphore(device, imageAvailableSem, nullptr);
        vkDestroySemaphore(device, renderFinishedSem, nullptr);
        vkDestroyFence(device, inFlightFence, nullptr);
    }
};

// ─────────────────────────────────────────────────────────────
// 应用层：VkApp
// ─────────────────────────────────────────────────────────────
class VkApp {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
    
private:
    VkContext context_;
    VkRenderFramework framework_;
    VkSwapChainManager swapchain_;
    std::vector<VkFrameResource> frameResources_;  // MAX_FRAMES_IN_FLIGHT 份
    uint32_t currentFrame_ = 0;
    
    void initVulkan() {
        context_.init("MyApp");
        framework_.init(context_, surface_);
        swapchain_.init(context_, framework_);
        
        // 创建帧资源池
        frameResources_.resize(MAX_FRAMES_IN_FLIGHT);
        for (auto& res : frameResources_) {
            res.init(context_.device(), framework_.commandPool());
        }
    }
    
    void drawFrame() {
        auto& frame = frameResources_[currentFrame_];
        
        // 等待上一帧完成
        vkWaitForFences(context_.device(), 1, &frame.inFlightFence, 
                       VK_TRUE, UINT64_MAX);
        
        // 获取图像
        uint32_t imageIdx;
        vkAcquireNextImageKHR(context_.device(), swapchain_.swapchain(),
                            UINT64_MAX, frame.imageAvailableSem, 
                            VK_NULL_HANDLE, &imageIdx);
        
        // 重置栅栏
        vkResetFences(context_.device(), 1, &frame.inFlightFence);
        
        // 录制命令
        recordCommandBuffer(frame.commandBuffer, imageIdx);
        
        // 提交
        submitFrame(frame);
        
        // 呈现
        presentFrame(imageIdx, frame);
        
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
    }
};
```

---

### 2.2 方案B：单体对象（简单项目可用）

```cpp
// 将所有Vulkan对象集中在一个 VkRenderer 类中
class VkRenderer {
public:
    void init(GLFWwindow* window) { /* 初始化所有 */ }
    void render() { /* 单帧渲染 */ }
    void cleanup() { /* 清理所有 */ }
    
private:
    // 层 1：上下文
    VkInstance instance_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    VkQueue graphicsQueue_, presentQueue_;
    
    // 层 2：渲染框架
    VkRenderPass renderPass_;
    VkPipeline pipeline_;
    VkPipelineLayout pipelineLayout_;
    VkCommandPool commandPool_;
    
    // 层 3：交换链
    VkSwapchainKHR swapchain_;
    std::vector<VkFramebuffer> framebuffers_;
    
    // 层 4：帧资源
    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence> inFlightFences_;
};
```

---

## 三、依赖关系梳理

### 3.1 创建顺序（必须遵守）

```
Instance → PhysicalDevice
   ↓           ↓
Surface ──────┘
   ↓
Device ← PhysicalDevice
   ↓
├─→ RenderPass
│   ├─→ Pipeline
│   └─→ Framebuffer ← ImageViews ← Swapchain
│
├─→ CommandPool → CommandBuffer
│
└─→ Semaphore + Fence
```

### 3.2 重建规则（窗口大小改变）

**需要重建的对象：**
- Swapchain
- ImageViews
- Framebuffers

**NOT 需要重建的对象：**
- Instance, Device, RenderPass, Pipeline, CommandPool

---

## 四、每帧同步流程

```
drawFrame() {
    // 1️⃣ 等待上一帧完成（GPU→CPU 同步）
    vkWaitForFences(device, 1, &inFlightFence[frame])
    
    // 2️⃣ 获取下一张图像
    uint32_t imageIdx
    vkAcquireNextImageKHR(..., &imageIdx)  
        // 返回后 imageAvailableSemaphore 被 signal
    
    // 3️⃣ 重置栅栏
    vkResetFences(device, 1, &inFlightFence[frame])
    
    // 4️⃣ 录制命令
    vkResetCommandBuffer(commandBuffer[frame])
    recordCommands(commandBuffer[frame], imageIdx)
    
    // 5️⃣ 提交到GPU（GPU→GPU 同步）
    VkSubmitInfo submit{
        .waitSemaphoreCount = 1
        .pWaitSemaphores = &imageAvailableSem[frame]
        .pWaitDstStageMask = &VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        .commandBufferCount = 1
        .pCommandBuffers = &commandBuffer[frame]
        .signalSemaphoreCount = 1
        .pSignalSemaphores = &renderFinishedSem[frame]
    }
    vkQueueSubmit(graphicsQueue, 1, &submit, inFlightFence[frame])
        // inFlightFence 在GPU完成后被 signal
    
    // 6️⃣ 呈现
    vkQueuePresentKHR(presentQueue, &present)
    
    // 7️⃣ 进入下一帧
    frame = (frame + 1) % MAX_FRAMES_IN_FLIGHT
}
```

---

## 五、实践建议

| 方面 | 建议 |
|------|------|
| **小项目** | 用方案B（单体对象），快速上手 |
| **中项目** | 用方案A（逐层封装），便于维护 |
| **大项目** | 方案A + 资源管理器 + 脚本化 |
| **多线程** | 每个线程独有 CommandBuffer + CommandPool，共享其他 |
| **异步** | 使用多个 Queue，Semaphore 链接先后关系 |

---

## 六、常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| ❌ Fence 永远不触发 | Fence 初始化时未设 SIGNALED_BIT | 加上 `VK_FENCE_CREATE_SIGNALED_BIT` |
| ❌ 内存泄漏 | 重建时没销毁旧对象 | 先 `destroySwapchain()` 再 `createSwapchain()` |
| ❌ 死锁 | vkWaitForFences 等不到 | 确认 vkQueueSubmit 有正确提交 |
| ❌ 花屏 | 栅栏时序错误 | 检查 waitStages 是否为 COLOR_ATTACHMENT_OUTPUT |

