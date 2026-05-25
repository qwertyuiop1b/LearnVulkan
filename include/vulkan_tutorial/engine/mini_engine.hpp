#pragma once
/**
 * @file mini_engine.hpp
 * @brief 第70章：迷你游戏引擎 —— 整合 ch61–ch69 的封装层
 *
 * MiniEngine 是一个轻量级的渲染引擎框架。
 * 它把 ch61–ch69 的各个模块组合在一起，
 * 提供一个面向"业务代码"的高层 API。
 *
 * 对比 ch60（裸 Vulkan，~1600 行）：
 * 使用 MiniEngine 的等价场景 Demo 只需约 150 行代码。
 *
 * 架构：
 *
 *   Application           ← 用户继承这个类
 *       ↓ createScene()   ← 用户填充场景
 *       ↓ onUpdate(dt)    ← 用户每帧更新逻辑
 *
 *   MiniEngine            ← 引擎主类（非继承，组合模式）
 *       ├─ RHIDevice
 *       ├─ Swapchain
 *       ├─ TextureCache
 *       ├─ ShaderLibrary
 *       ├─ MaterialLibrary
 *       ├─ World (ECS)
 *       ├─ FrustumCuller
 *       ├─ DrawCallBatch
 *       └─ FrameGraph（来自 ch51 的 RenderGraph）
 */

#include "rhi_device.hpp"
#include <vulkan_tutorial/vk_helpers.hpp>   // DepthResources など
#include "rhi_buffer.hpp"
#include "rhi_texture.hpp"
#include "rhi_shader.hpp"
#include "pipeline_builder.hpp"
#include "descriptor_manager.hpp"
#include "command_recorder.hpp"
#include "scene_ecs.hpp"
#include "material_system.hpp"

#include <vulkan_tutorial/render_graph.hpp>   // ch51 的 RenderGraph
#include <GLFW/glfw3.h>
#include <functional>
#include <memory>
#include <string>

namespace engine {

// ─── 引擎配置 ─────────────────────────────────────────────────────────────

struct EngineConfig {
    std::string appName   = "MiniEngine App";
    uint32_t    width     = 1024;
    uint32_t    height    = 768;
    uint32_t    frameCount = 2;
    bool        enableValidation = true;
    bool        vsync      = true;
    std::string shaderDir  = "shaders";     ///< .spv 着色器目录
    std::string assetDir   = "assets";      ///< 纹理/模型根目录
};

// ─── 帧上下文（每帧传递给渲染代码）────────────────────────────────────────

struct FrameContext {
    VkCommandBuffer cmd        = VK_NULL_HANDLE;
    uint32_t        frameIndex = 0;
    uint32_t        imageIndex = 0;
    float           deltaTime  = 0.016f;
    float           totalTime  = 0.0f;
    VkExtent2D      extent{};
};

// ─── Application 基类（用户继承）──────────────────────────────────────────

/**
 * @brief 用户继承的应用基类
 *
 * 最简用法：
 * @code
 *   class MyGame : public Application {
 *   public:
 *       void onInit(MiniEngine& eng) override
 *       {
 *           eng.textures().load("assets/ground.png");
 *           auto ground = eng.world().createEntity("Ground");
 *           eng.world().add<TransformComponent>(ground, {});
 *           eng.world().add<MeshComponent>(ground, buildGroundMesh());
 *       }
 *       void onUpdate(MiniEngine& eng, float dt) override
 *       {
 *           // 更新游戏逻辑
 *       }
 *   };
 *
 *   int main() {
 *       MiniEngine engine;
 *       engine.run(EngineConfig{}, MyGame{});
 *   }
 * @endcode
 */
class Application {
public:
    virtual ~Application() = default;
    virtual void onInit  (class MiniEngine& engine)             {}
    virtual void onUpdate(class MiniEngine& engine, float dt)   {}
    virtual void onRender(class MiniEngine& engine,
                          const FrameContext& ctx)               {}
    virtual void onResize(class MiniEngine& engine,
                          uint32_t w, uint32_t h)               {}
    virtual void onShutdown(class MiniEngine& engine)           {}

    /// 返回 false 时退出主循环
    virtual bool onEvent(int key, int action)
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) return false;
        return true;
    }
};

// ─── MiniEngine ───────────────────────────────────────────────────────────

class MiniEngine {
public:
    MiniEngine() = default;
    ~MiniEngine() { shutdown(); }
    MiniEngine(const MiniEngine&) = delete;
    MiniEngine& operator=(const MiniEngine&) = delete;

    /// 初始化引擎并进入主循环
    void run(const EngineConfig& config, Application& app);

    // ─── 访问子系统 ─────────────────────────────────────────────────────
    [[nodiscard]] RHIDevice&       device()    { return device_; }
    [[nodiscard]] World&           world()     { return world_; }
    [[nodiscard]] TextureCache&    textures()  { return textures_; }
    [[nodiscard]] ShaderLibrary&   shaders()   { return shaders_; }
    [[nodiscard]] MaterialLibrary& materials() { return materials_; }
    [[nodiscard]] PipelineCache&   pipelines() { return pipelineCache_; }
    [[nodiscard]] DrawCallBatch&   drawBatch() { return drawBatch_; }
    [[nodiscard]] vulkan_tutorial::RenderGraph& renderGraph() { return renderGraph_; }

    [[nodiscard]] VkRenderPass     mainRenderPass() const { return mainRP_; }
    [[nodiscard]] VkExtent2D       extent()          const { return extent_; }
    [[nodiscard]] uint32_t         frameCount()      const { return config_.frameCount; }
    [[nodiscard]] GLFWwindow*      window()           const { return window_; }

    // ─── 工具方法 ────────────────────────────────────────────────────────
    /// 快速提交一个 Mesh 到绘制队列（内部分配描述符集、绑定材质）
    void submit(EntityID entity,
                VkPipeline pipeline,
                VkDescriptorSet descSet,
                DrawKey::Layer layer = DrawKey::Opaque);

    /// 注册标准的前向渲染 Pass（适合大多数简单场景）
    void setupForwardPass(VkFormat colorFmt = VK_FORMAT_UNDEFINED,
                          bool withDepth = true, bool withBloom = false);

    /// 创建 Framebuffer（内部用，也开放给 Application）
    [[nodiscard]] VkFramebuffer createFramebuffer(VkRenderPass rp,
                                                   const std::vector<VkImageView>& views);

private:
    void init(const EngineConfig& config);
    void shutdown();
    void mainLoop(Application& app);
    void renderFrame(Application& app, float dt);
    void recreateSwapchain();

    // 交换链
    void createSwapchain();
    void createRenderPass();
    void createFramebuffers();
    void createSyncObjects();

    EngineConfig  config_{};
    GLFWwindow*   window_  = nullptr;

    // 子系统
    RHIDevice          device_;
    World              world_;
    TextureCache       textures_;
    ShaderLibrary      shaders_;
    MaterialLibrary    materials_;
    PipelineCache      pipelineCache_;
    DrawCallBatch      drawBatch_;
    FrustumCuller      culler_;
    vulkan_tutorial::RenderGraph renderGraph_;

    // 交换链资源
    VkSwapchainKHR   swapchain_ = VK_NULL_HANDLE;
    VkFormat         swapFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D       extent_{};
    std::vector<VkImage>       swapImages_;
    std::vector<VkImageView>   swapViews_;
    std::vector<VkFramebuffer> swapFBs_;
    VkRenderPass               mainRP_ = VK_NULL_HANDLE;
    vulkan_tutorial::DepthResources depth_{};

    // 同步对象
    std::vector<VkSemaphore> imgAvail_, renderDone_;
    std::vector<VkFence>     inFlight_;

    // 命令缓冲区
    CommandPool  cmdPool_;
    std::vector<VkCommandBuffer> cmdBufs_;

    uint32_t currentFrame_ = 0;
    float    totalTime_    = 0.0f;
    bool     resized_      = false;
public:
    void markResized() { resized_ = true; }   ///< 供 GLFW resize callback 调用
private:
};

} // namespace engine
