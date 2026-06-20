# Vulkan 三角形渲染架构 - 对比与最佳实践

## 一、原始代码 vs 重构代码

### 1.1 原始设计（ch08_triangle.cpp）

```
✅ 优点：
- 逻辑清晰，易于学习
- 代码行数少
- 每个步骤一目了然

❌ 缺点：
- 所有对象混在一个 Ch08App 类中（>1000 行）
- 重建交换链时需要手动清理多个对象
- 无法复用代码
- 难以单元测试
- 扩展功能需要修改主类
```

### 1.2 重构后设计（vk_renderer_refactored.hpp）

```
✅ 优点：
- 职责分明：每个类负责一种对象或概念
- 易于测试：每层可独立测试
- 易于复用：可以在多个项目中使用
- 易于扩展：添加新功能只需添加新类
- 易于维护：修改某层时不影响其他层

缺点：
- 代码行数更多
- 初学者需要理解分层架构
```

---

## 二、分层架构详解

### 2.1 层级关系

```
┌─────────────────────────────────┐
│  应用层（VkRenderer）            │
│  - 高层接口：init(), render()    │
│  - 管理生命周期                  │
└─────────────────────────────────┘
           ↓ 使用
┌─────────────────────────────────┐
│  层 1：VkContext                │
│  - 管理 Instance/Device         │
│  - 创建 CommandPool             │
│  - 不变的基础设施               │
└─────────────────────────────────┘
           ↓ 使用
┌─────────────────────────────────┐
│  层 2：VkRenderFramework         │
│  - RenderPass + Pipeline        │
│  - 定义"如何渲染"               │
│  - 不变的渲染配置               │
└─────────────────────────────────┘
           ↓ 使用
┌─────────────────────────────────┐
│  层 3：VkSwapChainManager        │
│  - 管理 Swapchain + 相关对象    │
│  - 支持重建（窗口大小改变）     │
│  - 变化的屏幕相关资源           │
└─────────────────────────────────┘
           ↓ 使用
┌─────────────────────────────────┐
│  层 4：VkFrameResource           │
│  - CommandBuffer + 同步原语      │
│  - 每帧循环使用                  │
└─────────────────────────────────┘
```

---

## 三、关键设计模式

### 3.1 单一职责原则（SRP）

```cpp
// ❌ 不好：所有职责混在一个类中
class BadApp {
    void createInstance() { ... }
    void createDevice() { ... }
    void createSwapchain() { ... }
    void recordCommandBuffer() { ... }
    void render() { ... }
    // 1000+ 行代码...
};

// ✅ 好：每个类负责一个职责
class VkContext {
    void createInstance() { ... }  // 只管 Instance/Device
};

class VkRenderFramework {
    void createRenderPass() { ... }  // 只管 RenderPass/Pipeline
};

class VkSwapChainManager {
    void createSwapchain() { ... }  // 只管 Swapchain 及其重建
};

class VkRenderer {
    void render() { ... }  // 只管应用逻辑
};
```

### 3.2 依赖注入

```cpp
// ❌ 不好：紧耦合
class VkRenderFramework {
    VkContext context_;  // 无法替换，无法测试
};

// ✅ 好：松耦合
class VkRenderFramework {
    std::shared_ptr<VkContext> context_;  // 可以传入任何实现
    
    void init(std::shared_ptr<VkContext> context) {
        context_ = context;
    }
};
```

### 3.3 资源生命周期管理

```cpp
// VkFrameResource 使用 RAII 思想
struct VkFrameResource {
    void init(VkDevice device, VkCommandPool pool) {
        // 构造时分配资源
        vkCreateSemaphore(..., &imageAvailableSem);
        vkCreateFence(..., &inFlightFence);
    }
    
    void cleanup(VkDevice device) {
        // 析构时释放资源
        vkDestroySemaphore(device, imageAvailableSem, nullptr);
        vkDestroyFence(device, inFlightFence, nullptr);
    }
};
```

---

## 四、渲染流程对比

### 原始代码

```cpp
void mainLoop() {
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        drawFrame();  // 一个函数处理所有逻辑
    }
}

void drawFrame() {
    // 100+ 行代码...
    vkWaitForFences(...);
    vkAcquireNextImageKHR(...);
    vkResetFences(...);
    recordCommandBuffer(...);
    vkQueueSubmit(...);
    vkQueuePresentKHR(...);
}
```

### 重构后代码

```cpp
void render() {
    auto& frame = frameResources_[currentFrame_];
    
    // 清晰的步骤分解
    vkWaitForFences(...);
    uint32_t imageIndex = acquireNextImage(frame);
    vkResetFences(...);
    recordCommandBuffer(frame.commandBuffer, imageIndex);
    submitFrame(frame);
    presentFrame(imageIndex, frame);
    
    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

private:
    void recordCommandBuffer(...) { ... }  // 独立函数
    void submitFrame(...) { ... }         // 独立函数
    void presentFrame(...) { ... }        // 独立函数
```

---

## 五、交换链重建处理

### 原始代码

```cpp
void recreateSwapchain() {
    // ... 等待设备空闲 ...
    cleanupSwapchain();
    createSwapchain();
    createImageViews();
    createFramebuffers();
    // 需要手动管理多个对象的清理/创建顺序
}

void cleanupSwapchain() {
    for (auto fb : framebuffers_) vkDestroyFramebuffer(...);
    for (auto iv : swapchainImageViews_) vkDestroyImageView(...);
    vkDestroySwapchainKHR(...);
}
```

### 重构后代码

```cpp
class VkSwapChainManager {
    void recreate() {
        // 等待设备空闲
        vkDeviceWaitIdle(context_->device());
        
        // 委托给内部函数处理细节
        destroySwapchain();  // 一个函数统一清理所有对象
        createSwapchain();
        createImageViews();
        createFramebuffers();
    }
    
private:
    void destroySwapchain() {
        for (auto fb : framebuffers_) vkDestroyFramebuffer(...);
        for (auto iv : imageViews_) vkDestroyImageView(...);
        if (swapchain_) vkDestroySwapchainKHR(...);
    }
};

// 使用时简单得多
renderer.onWindowResized();  // 只需调用一个函数
```

---

## 六、错误处理与验证

### 原始代码中的问题

```cpp
// 混在一起，难以定位错误
void drawFrame() {
    vkWaitForFences(...);
    uint32_t imageIdx;
    VkResult result = vkAcquireNextImageKHR(...);
    
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    // ... 继续处理，但错误可能来自任何地方
}
```

### 重构后的改进

```cpp
void render() {
    auto& frame = frameResources_[currentFrame_];
    
    vkWaitForFences(...);
    
    uint32_t imageIndex;
    VkResult result = acquireNextImage(frame, &imageIndex);
    
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchain_->recreate();
        return;
    }
    
    // 每一步都可以独立验证和调试
    recordCommandBuffer(frame.commandBuffer, imageIndex);
    submitFrame(frame);
    presentFrame(imageIndex, frame);
    
    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}
```

---

## 七、扩展性对比

### 添加新功能：纹理采样

**原始代码：** 需要修改主类，添加大量代码到 Ch08App 中

**重构后代码：** 可以创建新类 `VkTextureManager`

```cpp
class VkTextureManager {
    void init(std::shared_ptr<VkContext> context) { ... }
    void loadTexture(const std::string& path) { ... }
    void cleanup() { ... }
};

class VkRenderer {
    std::shared_ptr<VkTextureManager> textureManager_;
    
    void init(GLFWwindow* window) {
        // ... 其他初始化 ...
        textureManager_ = std::make_shared<VkTextureManager>();
        textureManager_->init(context_);
    }
};
```

---

## 八、性能对比

| 方面 | 原始代码 | 重构代码 |
|------|--------|--------|
| **初始化时间** | 快（代码少） | 略慢（多层调用） |
| **运行时性能** | 相同（最后都是同样的 Vulkan 调用） | 相同 |
| **内存占用** | 相同（对象数量相同） | 相同 |
| **缓存效率** | 相同 | 相同 |

💡 **结论：** 重构不会影响运行时性能，但提升代码质量和可维护性。

---

## 九、推荐的项目结构

```
project/
├── src/
│   ├── main.cpp                 # 入口点
│   ├── vk/
│   │   ├── context.hpp          # 第 1 层：VkContext
│   │   ├── framework.hpp        # 第 2 层：VkRenderFramework
│   │   ├── swapchain.hpp        # 第 3 层：VkSwapChainManager
│   │   ├── frame_resource.hpp   # 第 4 层：VkFrameResource
│   │   ├── renderer.hpp         # 顶层：VkRenderer
│   │   └── utils.hpp            # 共用工具函数
│   └── scene/
│       ├── mesh.hpp
│       ├── material.hpp
│       └── scene.hpp
├── shaders/
│   ├── triangle.vert
│   └── triangle.frag
└── CMakeLists.txt
```

---

## 十、学习路径

1. **初学者：** 学习原始代码（ch08_triangle.cpp）
   - 理解每个 Vulkan 对象的作用
   - 理解渲染循环流程

2. **进阶：** 学习重构代码（vk_renderer_refactored.hpp）
   - 理解分层架构
   - 理解设计模式（SRP、依赖注入）

3. **高级：** 自己重构项目
   - 添加新功能（纹理、模型、光照等）
   - 实现自己的 Vulkan 引擎框架

