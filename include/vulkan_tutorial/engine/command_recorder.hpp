#pragma once
/**
 * @file command_recorder.hpp
 * @brief 第67章：命令录制封装
 *
 * 封装层次：
 *   CommandPool       — 线程安全的 Pool 包装（每线程一个实例）
 *   CommandRecorder   — begin/end RAII 包装，自动处理 ONE_TIME_SUBMIT
 *   RenderPassScope   — begin/end render pass RAII 包装
 *   BarrierBatch      — 收集多个 barrier，一次 vkCmdPipelineBarrier 提交
 *   DrawCall / DrawKey — 排序友好的 draw call 数据结构
 */

#include "rhi_device.hpp"
#include <functional>
#include <vector>

namespace engine {

// ─── CommandPool ──────────────────────────────────────────────────────────

class CommandPool {
public:
    void create(RHIDevice& dev, uint32_t frameCount = 2);
    void destroy();
    void reset(uint32_t frameIndex);   ///< 帧开始时重置当帧的 pool

    [[nodiscard]] VkCommandBuffer allocate(uint32_t frameIndex,
                                           VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    [[nodiscard]] VkCommandPool pool(uint32_t frameIndex) const { return pools_[frameIndex]; }

private:
    RHIDevice*              dev_ = nullptr;
    std::vector<VkCommandPool> pools_;
};

// ─── CommandRecorder ──────────────────────────────────────────────────────

/**
 * @brief RAII 命令录制器
 *
 * 构造时 vkBeginCommandBuffer，析构（或 end()）时 vkEndCommandBuffer。
 *
 * 使用示例：
 * @code
 *   {
 *       CommandRecorder rec(dev, cmdBuf, CommandRecorder::OneShot);
 *       rec.cmd().bindPipeline(...);
 *       rec.cmd().draw(...);
 *   } // 自动 end
 * @endcode
 */
class CommandRecorder {
public:
    enum Hint { Normal, OneShot, ReRecordable };

    CommandRecorder() = default;
    CommandRecorder(RHIDevice& dev, VkCommandBuffer cmd, Hint hint = Normal);
    ~CommandRecorder();

    CommandRecorder(const CommandRecorder&) = delete;
    CommandRecorder& operator=(const CommandRecorder&) = delete;

    /// 显式结束（析构时会检查是否已 end）
    void end();

    [[nodiscard]] VkCommandBuffer handle() const { return cmd_; }

    /// 便利：直接在 recorder 上调用 draw、bind 等命令
    void bindPipeline(VkPipeline pipeline, VkPipelineBindPoint bp = VK_PIPELINE_BIND_POINT_GRAPHICS);
    void setViewportScissor(float x, float y, float w, float h);
    void pushConstants(VkPipelineLayout layout, VkShaderStageFlags stages,
                       uint32_t offset, uint32_t size, const void* data);

private:
    VkCommandBuffer cmd_    = VK_NULL_HANDLE;
    bool            ended_  = false;
};

// ─── RenderPassScope ──────────────────────────────────────────────────────

/**
 * @brief RAII render pass 范围
 *
 * 使用示例：
 * @code
 *   {
 *       RenderPassScope rp(cmd, sceneRP, fb, extent, clearValues);
 *       vkCmdDraw(cmd, ...);
 *   } // 自动 vkCmdEndRenderPass
 * @endcode
 */
class RenderPassScope {
public:
    RenderPassScope() = default;
    RenderPassScope(VkCommandBuffer cmd,
                    VkRenderPass renderPass,
                    VkFramebuffer framebuffer,
                    VkExtent2D extent,
                    std::initializer_list<VkClearValue> clearValues,
                    VkSubpassContents contents = VK_SUBPASS_CONTENTS_INLINE);
    ~RenderPassScope();

    RenderPassScope(const RenderPassScope&) = delete;
    RenderPassScope& operator=(const RenderPassScope&) = delete;
    RenderPassScope(RenderPassScope&&) noexcept;
    RenderPassScope& operator=(RenderPassScope&&) noexcept;

    void end();

private:
    VkCommandBuffer cmd_   = VK_NULL_HANDLE;
    bool            ended_ = false;
};

// ─── BarrierBatch ─────────────────────────────────────────────────────────

/**
 * @brief 批量 Pipeline Barrier 管理器
 *
 * 问题：每次 layout transition 都发出一个 vkCmdPipelineBarrier 效率低。
 * BarrierBatch：收集多个 barrier，一次 flush() 统一提交。
 *
 * 使用示例：
 * @code
 *   BarrierBatch batch(cmd);
 *   batch.imageLayout(hdrImage, VK_IMAGE_LAYOUT_UNDEFINED,
 *                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
 *                     VK_IMAGE_ASPECT_COLOR_BIT);
 *   batch.imageLayout(depthImage, VK_IMAGE_LAYOUT_UNDEFINED,
 *                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
 *                     VK_IMAGE_ASPECT_DEPTH_BIT);
 *   batch.flush();   // 一次 vkCmdPipelineBarrier 搞定两个转换
 * @endcode
 */
class BarrierBatch {
public:
    explicit BarrierBatch(VkCommandBuffer cmd) : cmd_(cmd) {}
    ~BarrierBatch() { if (!flushed_) flush(); }

    BarrierBatch& imageLayout(VkImage image,
                              VkImageLayout oldLayout,
                              VkImageLayout newLayout,
                              VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                              uint32_t mipLevels  = 1,
                              uint32_t layerCount = 1);

    BarrierBatch& buffer(VkBuffer buf,
                         VkAccessFlags srcAccess,
                         VkAccessFlags dstAccess,
                         VkDeviceSize  offset = 0,
                         VkDeviceSize  size   = VK_WHOLE_SIZE);

    /// 提交所有收集的 barrier（自动推导 stage flags）
    void flush();

private:
    VkCommandBuffer cmd_    = VK_NULL_HANDLE;
    bool            flushed_= false;
    std::vector<VkImageMemoryBarrier>  imageBarriers_;
    std::vector<VkBufferMemoryBarrier> bufferBarriers_;
    VkPipelineStageFlags srcStage_ = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStage_ = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
};

// ─── DrawCall 排序 ────────────────────────────────────────────────────────

/**
 * @brief 可排序的 Draw Call 描述符
 *
 * 游戏引擎通常按 (pipeline, material, mesh) 顺序排序 draw call，
 * 减少状态切换次数，提升 GPU 效率。
 *
 * DrawKey 是一个 64-bit 整数，按以下优先级编码：
 *   [63:56] 渲染层（透明 / 不透明 / UI）
 *   [55:40] Pipeline ID（减少管线切换）
 *   [39:24] Material ID（减少描述符切换）
 *   [23: 0] Mesh ID
 */
struct DrawKey {
    enum Layer : uint8_t { Opaque = 0, AlphaTest = 1, Transparent = 2, UI = 3 };

    static DrawKey make(Layer layer, uint16_t pipelineId,
                        uint16_t materialId, uint32_t meshId)
    {
        DrawKey k{};
        k.value = (uint64_t(layer)     << 56)
                | (uint64_t(pipelineId) << 40)
                | (uint64_t(materialId) << 24)
                | uint64_t(meshId & 0x00FFFFFFu);
        return k;
    }
    bool operator<(const DrawKey& o) const { return value < o.value; }
    uint64_t value = 0;
};

struct DrawCall {
    DrawKey             key;
    VkPipeline          pipeline  = VK_NULL_HANDLE;
    VkPipelineLayout    layout    = VK_NULL_HANDLE;
    VkDescriptorSet     descSet   = VK_NULL_HANDLE;
    VkBuffer            vertexBuf = VK_NULL_HANDLE;
    VkBuffer            indexBuf  = VK_NULL_HANDLE;
    uint32_t            indexCount  = 0;
    uint32_t            vertexCount = 0;
    uint32_t            instanceCount = 1;
    std::vector<uint8_t> pushConstantData;

    bool operator<(const DrawCall& o) const { return key < o.key; }
};

/// 收集并排序后提交 draw calls
class DrawCallBatch {
public:
    void add(DrawCall dc) { drawCalls_.push_back(std::move(dc)); }
    void sort()  { std::sort(drawCalls_.begin(), drawCalls_.end()); }
    void flush(VkCommandBuffer cmd);
    void clear() { drawCalls_.clear(); }

    [[nodiscard]] size_t count() const { return drawCalls_.size(); }

private:
    std::vector<DrawCall> drawCalls_;
    VkPipeline            lastPipeline_ = VK_NULL_HANDLE;
    VkDescriptorSet       lastDescSet_  = VK_NULL_HANDLE;
};

} // namespace engine
