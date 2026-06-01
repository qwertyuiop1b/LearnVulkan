#pragma once

/**
 * @file render_graph.hpp
 * @brief 简化版渲染图（Render Graph）
 *
 * 核心思想：
 *   - 用 Pass 描述渲染步骤，而不直接操作 Vulkan 命令
 *   - 根据 Pass 的输入/输出关系，自动推导并插入 Image Layout 屏障
 *   - 拥有并管理所有"瞬态"纹理（Pass 间传递的中间结果）
 *
 * 这不是工业级 Render Graph（无 DAG 分析、无别名、无异步计算），
 * 而是教学用途的最小实现，让你体会"声明式 vs 命令式"的区别。
 *
 * 用法示例：
 * @code
 *   RenderGraph graph;
 *   auto hdr    = graph.declareTexture({"hdr", VK_FORMAT_R16G16B16A16_SFLOAT, ...});
 *   auto bright = graph.declareTexture({"bright", VK_FORMAT_R16G16B16A16_SFLOAT, ...});
 *
 *   graph.addGraphicsPass("scene", {}, {hdr, bright}, {},
 *       [&](VkCommandBuffer cmd, uint32_t fi) { ... });
 *
 *   graph.addComputePass("blur", {bright}, {blurTex},
 *       [&](VkCommandBuffer cmd, uint32_t fi) { ... });
 *
 *   graph.build(device, physDev, cmdPool, queue, extent);
 *   // 每帧：
 *   graph.execute(cmd, frameIndex);
 * @endcode
 */

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vulkan_tutorial {

/// 瞬态纹理句柄（渲染图内部资源的引用）
struct RgTextureHandle {
    uint32_t id = UINT32_MAX;
    [[nodiscard]] bool isValid() const {
        return id != UINT32_MAX;
    }
};

/// 瞬态纹理描述符
struct RgTextureDesc {
    std::string name;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkExtent2D extent = {0, 0}; ///< 0×0 = 跟随交换链大小
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
};

/// Graphics Pass 执行回调
using GraphicsPassFn = std::function<void(VkCommandBuffer cmd, uint32_t frameIndex)>;

/// Compute Pass 执行回调
using ComputePassFn = std::function<void(VkCommandBuffer cmd, uint32_t frameIndex)>;

/**
 * @brief 简化版渲染图
 *
 * 负责：瞬态纹理的生命周期管理 + 自动 Image Layout 屏障推导。
 * 不负责：Framebuffer / RenderPass 创建（由 Pass 回调自行管理）。
 */
class RenderGraph {
  public:
    /// 声明一个瞬态纹理，返回其句柄
    RgTextureHandle declareTexture(const RgTextureDesc& desc);

    /**
     * @brief 添加一个图形 Pass
     * @param name        调试名称
     * @param reads       本 Pass 以采样方式读取的纹理
     * @param colorWrites 本 Pass 写入的颜色 Attachment 纹理
     * @param depthWrite  本 Pass 写入的深度 Attachment（可无效）
     * @param fn          实际录制 Vulkan 命令的回调（需自行调 vkCmdBeginRenderPass）
     */
    void addGraphicsPass(const char* name,
                         std::vector<RgTextureHandle> reads,
                         std::vector<RgTextureHandle> colorWrites,
                         RgTextureHandle depthWrite,
                         GraphicsPassFn fn);

    /**
     * @brief 添加一个计算 Pass
     * @param name   调试名称
     * @param reads  以 GENERAL 布局读取的纹理（Storage Image）
     * @param writes 以 GENERAL 布局写入的纹理
     * @param fn     实际录制 vkCmdDispatch 的回调
     */
    void addComputePass(const char* name,
                        std::vector<RgTextureHandle> reads,
                        std::vector<RgTextureHandle> writes,
                        ComputePassFn fn);

    /**
     * @brief 分配所有瞬态纹理
     * @param extent 若 RgTextureDesc.extent 为 {0,0}，则使用此值
     */
    void
    build(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool cmdPool, VkQueue queue, VkExtent2D extent);

    /// 交换链大小变化时重新分配瞬态纹理
    void resize(
        VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool cmdPool, VkQueue queue, VkExtent2D newExtent);

    /// 录制当前帧的所有 Pass（自动插入 Layout 屏障）
    void execute(VkCommandBuffer cmd, uint32_t frameIndex);

    /// 销毁所有瞬态纹理（在 vkDestroyDevice 之前调用）
    void destroy(VkDevice device);

    /// 清空所有 Pass 定义（纹理声明保留）
    void resetPasses();

    [[nodiscard]] VkImageView getView(RgTextureHandle h) const;
    [[nodiscard]] VkImage getImage(RgTextureHandle h) const;
    [[nodiscard]] VkImageLayout getLayout(RgTextureHandle h) const;

    /// 通知 RenderGraph 某个资源当前所在的 Layout（用于手动 Barrier 之后同步追踪状态）
    void setLayout(RgTextureHandle h, VkImageLayout layout);

    [[nodiscard]] bool isBuilt() const {
        return built_;
    }

  private:
    struct RgTexture {
        RgTextureDesc desc;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    enum class PassType { Graphics, Compute };

    struct PassNode {
        PassType type;
        std::string name;
        std::vector<RgTextureHandle> reads;
        std::vector<RgTextureHandle> colorWrites;
        RgTextureHandle depthWrite;
        GraphicsPassFn graphicsFn;
        ComputePassFn computeFn;
    };

    std::vector<RgTexture> textures_;
    std::vector<PassNode> passes_;
    VkExtent2D buildExtent_{};
    bool built_ = false;

    void allocateTextures(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool cmdPool, VkQueue queue);
    void freeTextures(VkDevice device);
    void transitionImage(VkCommandBuffer cmd, RgTexture& tex, VkImageLayout target);
};

} // namespace vulkan_tutorial
