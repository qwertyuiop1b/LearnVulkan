#pragma once
/**
 * @file pipeline_builder.hpp
 * @brief 第65章：管线构建器（流式 API）
 *
 * VkGraphicsPipelineCreateInfo 有 30+ 个字段，每次从零填写既繁琐又容易出错。
 * PipelineBuilder 提供：
 *   1. 流式 API（builder pattern）：每个方法返回 *this 供链式调用
 *   2. 合理的默认值（覆盖 95% 的使用场景）
 *   3. PipelineCache：哈希管线状态 → 跳过重复创建 + 序列化到磁盘
 *
 * 对比：
 *   【直接写法】~80 行 struct 初始化代码
 *   【PipelineBuilder】~5 行流式调用
 */

#include "rhi_device.hpp"
#include "rhi_shader.hpp"
#include <array>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

// ─── 管线状态描述（可哈希）────────────────────────────────────────────────

/// 顶点输入描述（聚合绑定+属性，避免悬垂指针）
struct VertexInputState {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
};

// ─── 图形管线构建器 ────────────────────────────────────────────────────────

/**
 * @brief 图形管线流式构建器
 *
 * 使用示例：
 * @code
 *   VkPipeline pipe = GraphicsPipelineBuilder(dev)
 *       .setProgram(shaderProgram)
 *       .setVertexInput<MyVertex>()
 *       .setDepthTest(true, true)
 *       .setCullMode(VK_CULL_MODE_BACK_BIT)
 *       .setAlphaBlend(false)
 *       .setRenderPass(renderPass, 0)
 *       .build(pipelineCache);
 * @endcode
 */
class GraphicsPipelineBuilder {
  public:
    explicit GraphicsPipelineBuilder(RHIDevice& dev) : dev_(&dev) {
        setDefaults();
    }

    // ── 着色器 ──────────────────────────────────────────────────────────
    GraphicsPipelineBuilder& setProgram(const ShaderProgram& prog);
    GraphicsPipelineBuilder& setLayout(VkPipelineLayout layout);

    // ── 顶点输入 ────────────────────────────────────────────────────────
    GraphicsPipelineBuilder& setVertexInput(const VertexInputState& vis);

    template <typename T> GraphicsPipelineBuilder& setVertexInput() {
        return setVertexInput(T::vertexInputState());
    }

    GraphicsPipelineBuilder& setNoVertexInput(); ///< 全屏三角形等无需顶点

    // ── 拓扑 & 光栅化 ───────────────────────────────────────────────────
    GraphicsPipelineBuilder& setTopology(VkPrimitiveTopology t);
    GraphicsPipelineBuilder& setCullMode(VkCullModeFlags cull, VkFrontFace front = VK_FRONT_FACE_COUNTER_CLOCKWISE);
    GraphicsPipelineBuilder& setPolygonMode(VkPolygonMode mode);
    GraphicsPipelineBuilder& setLineWidth(float w);
    GraphicsPipelineBuilder& setDepthBias(float constant, float slope);

    // ── 深度 & 模板 ─────────────────────────────────────────────────────
    GraphicsPipelineBuilder& setDepthTest(bool test, bool write = true, VkCompareOp op = VK_COMPARE_OP_LESS);
    GraphicsPipelineBuilder& setDepthBounds(float min, float max);
    GraphicsPipelineBuilder& setStencilTest(VkStencilOpState front, VkStencilOpState back);

    // ── 混合 ────────────────────────────────────────────────────────────
    GraphicsPipelineBuilder& setAlphaBlend(bool enable,
                                           VkBlendFactor srcColor = VK_BLEND_FACTOR_SRC_ALPHA,
                                           VkBlendFactor dstColor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                           VkBlendFactor srcAlpha = VK_BLEND_FACTOR_ONE,
                                           VkBlendFactor dstAlpha = VK_BLEND_FACTOR_ZERO);
    GraphicsPipelineBuilder& setColorWriteMask(VkColorComponentFlags mask);
    GraphicsPipelineBuilder& addColorAttachment(VkFormat fmt); ///< Dynamic Rendering

    // ── 多重采样 ─────────────────────────────────────────────────────────
    GraphicsPipelineBuilder& setMSAA(VkSampleCountFlagBits samples);

    // ── 动态状态 ─────────────────────────────────────────────────────────
    GraphicsPipelineBuilder& addDynamicState(VkDynamicState state);

    // ── RenderPass ───────────────────────────────────────────────────────
    GraphicsPipelineBuilder& setRenderPass(VkRenderPass rp, uint32_t subpass = 0);

    // ── 构建 ─────────────────────────────────────────────────────────────
    [[nodiscard]] VkPipeline build(VkPipelineCache cache = VK_NULL_HANDLE);

  private:
    void setDefaults();

    RHIDevice* dev_ = nullptr;

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    uint32_t subpass_ = 0;

    VertexInputState vertexInput_;
    VkPipelineVertexInputStateCreateInfo viState_{};
    VkPipelineInputAssemblyStateCreateInfo iaState_{};
    VkPipelineRasterizationStateCreateInfo rsState_{};
    VkPipelineDepthStencilStateCreateInfo dsState_{};
    VkPipelineMultisampleStateCreateInfo msState_{};

    std::vector<VkPipelineColorBlendAttachmentState> blendAttachments_;
    std::vector<VkDynamicState> dynamicStates_;
    std::vector<VkFormat> colorAttachmentFmts_; // Dynamic Rendering
    VkFormat depthAttachmentFmt_ = VK_FORMAT_UNDEFINED;
};

// ─── 计算管线构建器 ────────────────────────────────────────────────────────

class ComputePipelineBuilder {
  public:
    explicit ComputePipelineBuilder(RHIDevice& dev) : dev_(&dev) {}

    ComputePipelineBuilder& setProgram(const ShaderProgram& prog);
    [[nodiscard]] VkPipeline build(VkPipelineCache cache = VK_NULL_HANDLE);

  private:
    RHIDevice* dev_ = nullptr;
    VkPipelineShaderStageCreateInfo stage_{};
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
};

// ─── 管线缓存 ──────────────────────────────────────────────────────────────

/**
 * @brief 管线缓存
 *
 * 两层缓存：
 *   1. VkPipelineCache（Vulkan 原生，跨 session 的 SPIR-V 编译缓存）
 *      → 序列化到 pipeline.cache，下次启动直接加载，避免重新编译 SPIR-V
 *   2. StateHash → VkPipeline（运行时缓存，避免同状态重复创建管线对象）
 *
 * 使用：
 * @code
 *   PipelineCache cache;
 *   cache.init(dev, "pipeline.cache");
 *   VkPipeline pipe = builder.build(cache.vkCache());
 *   cache.save();   // 程序退出前保存
 * @endcode
 */
class PipelineCache {
  public:
    void init(RHIDevice& dev, const std::string& cacheFile = "");
    void destroy();
    void save(); ///< 序列化到磁盘

    [[nodiscard]] VkPipelineCache vkCache() const {
        return vkCache_;
    }

    /// 运行时管线对象缓存（按名字或哈希）
    [[nodiscard]] VkPipeline find(size_t stateHash) const;
    void store(size_t stateHash, VkPipeline pipeline);

    [[nodiscard]] size_t cachedPipelineCount() const {
        return runtimeCache_.size();
    }
    [[nodiscard]] bool hasFile() const {
        return !cacheFile_.empty();
    }

  private:
    RHIDevice* dev_ = nullptr;
    VkPipelineCache vkCache_ = VK_NULL_HANDLE;
    std::string cacheFile_;
    std::unordered_map<size_t, VkPipeline> runtimeCache_;
};

} // namespace engine
