# Vulkan 教程章节目录（70章完整路线图）

## 第一部分：Vulkan 基础管线（ch01–ch14）

| 章节 | 主题 | 核心概念 |
|------|------|---------|
| ch01 | Instance 创建 | VkInstance / ValidationLayer / DebugCallback |
| ch02 | 物理设备 | VkPhysicalDevice / 队列族 / 设备特性查询 |
| ch03 | 逻辑设备 | VkDevice / VkQueue / 设备扩展 |
| ch04 | 交换链 | VkSwapchainKHR / Surface / PresentMode |
| ch05 | 渲染通道 | VkRenderPass / Attachment / Subpass |
| ch06 | 图形管线 | VkPipeline / ShaderModule / 固定功能阶段 |
| ch07 | 命令缓冲区 | VkCommandPool / VkCommandBuffer / 录制 |
| ch08 | 第一个三角形 | 绘制调用 / 帧同步 / Semaphore / Fence |
| ch09 | 顶点缓冲区 | VkBuffer / VkDeviceMemory / Staging Buffer |
| ch10 | 统一缓冲区 | UBO / DescriptorSet / PipelineLayout |
| ch11 | 纹理 | VkImage / VkImageView / VkSampler |
| ch12 | 深度缓冲 | Depth Attachment / DepthStencil Layout |
| ch13 | Mipmap | 生成 Mip / LOD / 各向异性过滤 |
| ch14 | MSAA | 多重采样 / Resolve / 质量 vs 性能 |

## 第二部分：Compute 与高级特性（ch15–ch25）

| 章节 | 主题 | 核心概念 |
|------|------|---------|
| ch15 | Compute Shader | VkPipeline（Compute）/ SSBO / Dispatch |
| ch16 | Dynamic Rendering | VK_KHR_dynamic_rendering / 免 RenderPass |
| ch17 | 实例化渲染 | Instanced Draw / 每实例属性 |
| ch18 | 阴影映射 | Shadow Map / 深度偏移 / PCF |
| ch19 | 延迟渲染 | G-Buffer / 多颜色 Attachment |
| ch20 | 光线追踪 | VK_KHR_ray_tracing / AS / SBT / GLSL RT |
| ch21 | Mesh Shader | VK_EXT_mesh_shader / Task+Mesh 流水线 |
| ch22 | Bindless 渲染 | 大 DescriptorArray / NonUniform 索引 |
| ch23 | GPU Driven | IndirectDraw / GPU Culling / DrawCount |
| ch24 | 后处理 | HDR / Bloom / ACES Tone Mapping |
| ch25 | 高级同步 | Timeline Semaphore / 多线程 CB 录制 |

## 第三部分：Vulkan 扩展与工具（ch26–ch37）

| 章节 | 主题 | 核心概念 |
|------|------|---------|
| ch26 | Query Pool | 时间戳查询 / Pipeline Statistics |
| ch27 | Tessellation | Tess Control / Evaluation Shader |
| ch28 | Subgroup | gl_SubgroupInvocation / WaveOps |
| ch29 | VRS | 可变速率着色 / ShadingRate Image |
| ch30 | VMA | Vulkan Memory Allocator 集成 |
| ch31 | Shader Object | VK_EXT_shader_object |
| ch32 | Sparse Resources | VkSparseImage / 虚拟纹理 |
| ch33 | Push Descriptors | VK_KHR_push_descriptor |
| ch34 | Geometry Shader | 点云 / 法线可视化 |
| ch35 | Multiview | VR 左右眼同时渲染 |
| ch36 | GPL | Graphics Pipeline Library |
| ch37 | Conditional Rendering | 遮挡查询驱动的 Predicated Draw |

## 第四部分：渲染效果（ch38–ch50）

| 章节 | 主题 | 核心概念 |
|------|------|---------|
| ch38 | Alpha 混合 | 透明物体排序 / 混合方程 |
| ch39 | 模板缓冲 | 描边 / 遮罩 |
| ch40 | 法线贴图 | TBN 矩阵 / 切线空间 |
| ch41 | 天空盒 | Cubemap / VK_IMAGE_VIEW_TYPE_CUBE |
| ch42 | IBL | 辐照度 / 预滤波 / BRDF LUT |
| ch43 | glTF 加载 | cgltf / 材质 / 动画数据 |
| ch44 | PBR | Cook-Torrance BRDF / 金属度/粗糙度 |
| ch45 | CSM | 级联阴影 / 视锥拆分 |
| ch46 | SSAO | G-Buffer / 半球采样 / 模糊 |
| ch47 | SSR | 屏幕空间反射 / 步进求交 |
| ch48 | OIT | Weighted Blended / 顺序无关透明 |
| ch49 | TAA | 时间抗锯齿 / 历史帧重投影 |
| ch50 | 骨骼动画 | Skinning / Joint 矩阵 / glTF 动画 |

## 第五部分：游戏 Demo 技法（ch51–ch60）

| 章节 | 主题 | 核心概念 |
|------|------|---------|
| ch51 | Render Graph | 声明式 Pass / 自动 Barrier |
| ch52 | GPU 粒子 | 发射器 / 重生 / Billboard |
| ch53 | 水面渲染 | 平面反射 RTT / 折射 / Fresnel |
| ch54 | 体积雾 | 深度重建 / 指数高度雾 / God Rays |
| ch55 | 屏幕空间贴花 | Fullscreen Overlay / AABB 检测 |
| ch56 | 异步纹理加载 | std::thread / Staging Pool |
| ch57 | 多线程命令录制 | Secondary CB / vkCmdExecuteCommands |
| ch58 | PCSS 软阴影 | Blocker Search / Penumbra 估算 |
| ch59 | 反射探针 | Cubemap 捕获 / Fresnel 混合 |
| ch60 | 户外综合 Demo | Shadow+Water+Particles+Bloom+Fog |

## 第六部分：引擎封装（ch61–ch70）

| 章节 | 主题 | 封装 API |
|------|------|---------|
| ch61 | RHIDevice | `dev.init(ci)` 5行 = ch01-ch03全部功能 |
| ch62 | 缓冲区封装 | `Buffer` / `VertexBuffer` / `UniformBuffer<T>` / `StagingPool` |
| ch63 | 纹理系统 | `Texture2D::loadFromFile()` / `TextureCache` / `RenderTarget` |
| ch64 | 着色器系统 | `ShaderProgram::link()` 自动推导 DSL / `ShaderLibrary` 热重载 |
| ch65 | 管线构建器 | `GraphicsPipelineBuilder` 流式 API / `PipelineCache` 磁盘序列化 |
| ch66 | 描述符管理 | `DescriptorAllocator` 自动扩容 / `DescriptorBuilder` 4行构建 |
| ch67 | 命令录制 | `CommandRecorder` RAII / `BarrierBatch` / `DrawCallBatch` 排序 |
| ch68 | 场景 ECS | `World` / `ComponentStorage<T>` SoA / `FrustumCuller` |
| ch69 | 材质系统 | `Material` 模板 / `MaterialInstance` 参数覆盖 / `MaterialLibrary` |
| ch70 | MiniEngine | `Application` 基类 / 整合所有子系统 / 代码量减少 80% |

---

## 构建方式

```bash
# 在工程根目录
mkdir -p build && cd build
cmake ..
cmake --build . -j4               # 全部章节

# 单独构建某章
cmake --build . --target ch70_mini_engine

# 运行
./src/ch70_mini_engine
```

## 引擎头文件位置

```
include/vulkan_tutorial/engine/
  rhi_device.hpp         # 设备抽象
  rhi_buffer.hpp         # 缓冲区 RAII
  rhi_texture.hpp        # 纹理系统
  rhi_shader.hpp         # 着色器 + 反射
  pipeline_builder.hpp   # 管线构建器
  descriptor_manager.hpp # 描述符管理
  command_recorder.hpp   # 命令录制
  scene_ecs.hpp          # ECS 场景图
  material_system.hpp    # 材质系统
  mini_engine.hpp        # 引擎主类
  demo_app.hpp           # Demo 基类（ch62-ch70 共用）
```

## 控制方式（ch38+ 均支持）

| 操作 | 效果 |
|------|------|
| 鼠标左键拖拽 | 旋转相机 |
| 鼠标右键拖拽 | 平移相机 |
| 滚轮 | 缩放 |
| WASD/QE | 移动目标点 |
| R | 重置相机 |
| ESC | 退出 |
| ImGui 面板 | 调节各章节参数 |
