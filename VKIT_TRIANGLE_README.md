# VkKit Triangle Example

这是一个使用VkKit库创建的简单三角形示例，展示了如何使用VkKit的RAII封装来快速创建Vulkan应用。

## 文件结构

- `src/chapters/vkit_triangle.cpp` - 主程序代码
- `shaders/vkit_triangle.vert` - 顶点着色器（顶点数据硬编码）
- `shaders/vkit_triangle.frag` - 片段着色器

## 特点

- **完全依赖VkKit库**：使用VkKit提供的RAII封装，无需手动管理Vulkan资源
- **Dynamic Rendering**：使用Vulkan 1.3的动态渲染，无需创建RenderPass
- **无顶点缓冲区**：顶点数据直接硬编码在着色器中
- **简洁的代码**：相比原始Vulkan API，代码量大幅减少

## 核心组件

### 1. VulkanContext
创建Vulkan实例、设备和队列：
```cpp
VulkanContextCreateInfo contextInfo{};
contextInfo.window = window_;
contextInfo.applicationName = "VkKit Triangle";
contextInfo.enableValidation = true;
contextInfo.requireDynamicRendering = true;
contextInfo.requireSynchronization2 = true;
VulkanContext context(contextInfo);
```

### 2. Swapchain
管理交换链和呈现图像：
```cpp
SwapchainCreateInfo swapchainInfo{};
swapchainInfo.desiredExtent = {WIDTH, HEIGHT};
Swapchain swapchain(context, swapchainInfo);
```

### 3. FrameScheduler
管理帧同步、命令缓冲区和动态渲染：
```cpp
FrameSchedulerCreateInfo schedulerInfo{};
schedulerInfo.swapchain = &swapchain;
schedulerInfo.framesInFlight = 2;
FrameScheduler frameScheduler(context, schedulerInfo);
```

### 4. GraphicsPipeline
创建图形管线：
```cpp
GraphicsPipelineCreateInfo pipelineInfo{};
pipelineInfo.layout = &pipelineLayout;
pipelineInfo.shaderStages = {
    {&vertexShader, vk::ShaderStageFlagBits::eVertex, "main"},
    {&fragmentShader, vk::ShaderStageFlagBits::eFragment, "main"},
};
pipelineInfo.colorAttachmentFormats = {swapchain.format()};
GraphicsPipeline pipeline(context, pipelineInfo);
```

## 渲染循环

```cpp
// 1. 开始帧
const FrameBeginResult beginResult = frameScheduler.beginFrame();

// 2. 开始动态渲染
DynamicRenderingInfo renderingInfo{};
renderingInfo.colorClearValue = {{0.0f, 0.0f, 0.0f, 1.0f}};
frameScheduler.beginDynamicRendering(renderingInfo);

// 3. 设置视口和裁剪
vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

// 4. 绑定管线并绘制
vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.nativeHandle());
vkCmdDraw(commandBuffer, 3, 1, 0, 0);

// 5. 结束渲染和帧
frameScheduler.endDynamicRendering();
frameScheduler.endFrame();
```

## 着色器

### 顶点着色器 (vkit_triangle.vert)
```glsl
#version 450

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),  // 红色
    vec3(0.0, 1.0, 0.0),  // 绿色
    vec3(0.0, 0.0, 1.0)   // 蓝色
);

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}
```

### 片段着色器 (vkit_triangle.frag)
```glsl
#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
```

## 编译和运行

```bash
# 配置CMake
cmake -B build -S .

# 编译
cmake --build build --target vkit_triangle

# 运行
./build/src/vkit_triangle
```

## 与传统Vulkan代码对比

使用VkKit库后，代码量大幅减少：

| 功能 | 传统Vulkan | VkKit |
|------|-----------|-------|
| 实例创建 | ~50行 | 3行 |
| 设备选择和创建 | ~100行 | 包含在VulkanContext |
| 交换链管理 | ~150行 | 3行 |
| 命令缓冲区管理 | ~80行 | 包含在FrameScheduler |
| 同步对象 | ~60行 | 包含在FrameScheduler |
| 资源销毁 | 手动逐个销毁 | RAII自动销毁 |

## 优势

1. **RAII资源管理**：无需手动调用vkDestroy*函数
2. **更少的样板代码**：封装了常见的Vulkan模式
3. **更安全**：资源生命周期由C++对象管理
4. **现代Vulkan**：默认使用Vulkan 1.3和动态渲染
5. **易于维护**：代码更简洁，更容易理解

## 注意事项

- VkKit对象不是线程安全的，需要外部同步
- 所有依赖VulkanContext的对象必须先于VulkanContext销毁
- 当前示例使用2个in-flight帧（可配置）
- 窗口调整大小时自动重建交换链

## 参考

- VkKit库文档：`lib/VkKit/README.md`
- Vulkan规范：https://www.khronos.org/vulkan/
