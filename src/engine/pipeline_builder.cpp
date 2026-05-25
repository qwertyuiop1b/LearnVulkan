/**
 * @file pipeline_builder.cpp
 * @brief 第65章：管线构建器实现
 *
 * 对比：
 *   传统方式 — ~80 行 struct 初始化，极易遗漏字段
 *   Builder  — ~5 行流式调用，默认值覆盖 95% 场景
 *
 * 实现细节：
 *   - setDefaults() 填充合理初始值（深度测试开启、背面剔除、动态 Viewport/Scissor）
 *   - build() 组装 VkGraphicsPipelineCreateInfo 并调用 vkCreateGraphicsPipelines
 *   - PipelineCache 加载/保存磁盘序列化缓存 + 运行时 hash → VkPipeline 映射
 */

#include <vulkan_tutorial/engine/pipeline_builder.hpp>
#include <vulkan_tutorial/utils.hpp>

#include <fstream>
#include <stdexcept>

namespace engine {

// ─── GraphicsPipelineBuilder：默认值 ─────────────────────────────────────────

void GraphicsPipelineBuilder::setDefaults()
{
    // 顶点输入（默认无输入，由 setVertexInput 覆盖）
    viState_.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    // 图元拓扑：三角形列表
    iaState_.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaState_.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    iaState_.primitiveRestartEnable = VK_FALSE;

    // 光栅化：背面剔除，逆时针为正面
    rsState_.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rsState_.depthClampEnable        = VK_FALSE;
    rsState_.rasterizerDiscardEnable = VK_FALSE;
    rsState_.polygonMode             = VK_POLYGON_MODE_FILL;
    rsState_.cullMode                = VK_CULL_MODE_BACK_BIT;
    rsState_.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsState_.depthBiasEnable         = VK_FALSE;
    rsState_.lineWidth               = 1.0f;

    // 深度/模板：开启深度测试和写入
    dsState_.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dsState_.depthTestEnable       = VK_TRUE;
    dsState_.depthWriteEnable      = VK_TRUE;
    dsState_.depthCompareOp        = VK_COMPARE_OP_LESS;
    dsState_.depthBoundsTestEnable = VK_FALSE;
    dsState_.stencilTestEnable     = VK_FALSE;

    // 多重采样：单采样
    msState_.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msState_.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    msState_.sampleShadingEnable  = VK_FALSE;

    // 默认颜色混合附件：无透明混合，写入 RGBA
    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAtt.blendEnable = VK_FALSE;
    blendAttachments_.push_back(blendAtt);

    // 动态状态：Viewport 和 Scissor（最常见的动态状态）
    addDynamicState(VK_DYNAMIC_STATE_VIEWPORT);
    addDynamicState(VK_DYNAMIC_STATE_SCISSOR);
}

// ─── 流式接口方法 ─────────────────────────────────────────────────────────────

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setProgram(const ShaderProgram& prog)
{
    shaderStages_ = prog.stageInfos();
    layout_       = prog.pipelineLayout();
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setLayout(VkPipelineLayout layout)
{
    layout_ = layout;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setVertexInput(const VertexInputState& vis)
{
    vertexInput_ = vis;
    viState_.vertexBindingDescriptionCount   = static_cast<uint32_t>(vertexInput_.bindings.size());
    viState_.pVertexBindingDescriptions      = vertexInput_.bindings.data();
    viState_.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInput_.attributes.size());
    viState_.pVertexAttributeDescriptions    = vertexInput_.attributes.data();
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setNoVertexInput()
{
    vertexInput_ = {};
    viState_.vertexBindingDescriptionCount   = 0;
    viState_.pVertexBindingDescriptions      = nullptr;
    viState_.vertexAttributeDescriptionCount = 0;
    viState_.pVertexAttributeDescriptions    = nullptr;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setTopology(VkPrimitiveTopology t)
{
    iaState_.topology = t;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setCullMode(VkCullModeFlags cull,
                                                               VkFrontFace front)
{
    rsState_.cullMode  = cull;
    rsState_.frontFace = front;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setPolygonMode(VkPolygonMode mode)
{
    rsState_.polygonMode = mode;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setLineWidth(float w)
{
    rsState_.lineWidth = w;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setDepthBias(float constant, float slope)
{
    rsState_.depthBiasEnable         = (constant != 0.0f || slope != 0.0f) ? VK_TRUE : VK_FALSE;
    rsState_.depthBiasConstantFactor = constant;
    rsState_.depthBiasSlopeFactor    = slope;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setDepthTest(bool test, bool write,
                                                                VkCompareOp op)
{
    dsState_.depthTestEnable  = test  ? VK_TRUE : VK_FALSE;
    dsState_.depthWriteEnable = write ? VK_TRUE : VK_FALSE;
    dsState_.depthCompareOp   = op;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setDepthBounds(float minVal, float maxVal)
{
    dsState_.depthBoundsTestEnable = VK_TRUE;
    dsState_.minDepthBounds        = minVal;
    dsState_.maxDepthBounds        = maxVal;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setStencilTest(VkStencilOpState front,
                                                                   VkStencilOpState back)
{
    dsState_.stencilTestEnable = VK_TRUE;
    dsState_.front             = front;
    dsState_.back              = back;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setAlphaBlend(
    bool enable,
    VkBlendFactor srcColor, VkBlendFactor dstColor,
    VkBlendFactor srcAlpha, VkBlendFactor dstAlpha)
{
    if (blendAttachments_.empty())
        blendAttachments_.resize(1);

    auto& att = blendAttachments_[0];
    att.blendEnable         = enable ? VK_TRUE : VK_FALSE;
    att.srcColorBlendFactor = srcColor;
    att.dstColorBlendFactor = dstColor;
    att.colorBlendOp        = VK_BLEND_OP_ADD;
    att.srcAlphaBlendFactor = srcAlpha;
    att.dstAlphaBlendFactor = dstAlpha;
    att.alphaBlendOp        = VK_BLEND_OP_ADD;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setColorWriteMask(VkColorComponentFlags mask)
{
    if (blendAttachments_.empty())
        blendAttachments_.resize(1);
    blendAttachments_[0].colorWriteMask = mask;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::addColorAttachment(VkFormat fmt)
{
    colorAttachmentFmts_.push_back(fmt);
    if (colorAttachmentFmts_.size() > blendAttachments_.size()) {
        VkPipelineColorBlendAttachmentState att{};
        att.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blendAttachments_.push_back(att);
    }
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setMSAA(VkSampleCountFlagBits samples)
{
    msState_.rasterizationSamples = samples;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::addDynamicState(VkDynamicState state)
{
    if (std::find(dynamicStates_.begin(), dynamicStates_.end(), state) == dynamicStates_.end())
        dynamicStates_.push_back(state);
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setRenderPass(VkRenderPass rp,
                                                                  uint32_t subpass)
{
    renderPass_ = rp;
    subpass_    = subpass;
    return *this;
}

// ─── GraphicsPipelineBuilder::build ─────────────────────────────────────────

VkPipeline GraphicsPipelineBuilder::build(VkPipelineCache cache)
{
    // 动态状态
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = static_cast<uint32_t>(dynamicStates_.size());
    dynState.pDynamicStates    = dynamicStates_.empty() ? nullptr : dynamicStates_.data();

    // Viewport（动态时仅设置 count）
    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1;
    vpState.scissorCount  = 1;

    // 颜色混合
    VkPipelineColorBlendStateCreateInfo blendState{};
    blendState.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendState.logicOpEnable   = VK_FALSE;
    blendState.attachmentCount = static_cast<uint32_t>(blendAttachments_.size());
    blendState.pAttachments    = blendAttachments_.empty() ? nullptr : blendAttachments_.data();

    // 组装 VkGraphicsPipelineCreateInfo
    VkGraphicsPipelineCreateInfo pipeCI{};
    pipeCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeCI.stageCount          = static_cast<uint32_t>(shaderStages_.size());
    pipeCI.pStages             = shaderStages_.empty() ? nullptr : shaderStages_.data();
    pipeCI.pVertexInputState   = &viState_;
    pipeCI.pInputAssemblyState = &iaState_;
    pipeCI.pViewportState      = &vpState;
    pipeCI.pRasterizationState = &rsState_;
    pipeCI.pMultisampleState   = &msState_;
    pipeCI.pDepthStencilState  = &dsState_;
    pipeCI.pColorBlendState    = &blendState;
    pipeCI.pDynamicState       = &dynState;
    pipeCI.layout              = layout_;
    pipeCI.renderPass          = renderPass_;
    pipeCI.subpass             = subpass_;
    pipeCI.basePipelineHandle  = VK_NULL_HANDLE;
    pipeCI.basePipelineIndex   = -1;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(dev_->device(), cache, 1, &pipeCI, nullptr, &pipeline));
    return pipeline;
}

// ─── ComputePipelineBuilder ──────────────────────────────────────────────────

ComputePipelineBuilder& ComputePipelineBuilder::setProgram(const ShaderProgram& prog)
{
    const auto& stages = prog.stageInfos();
    if (!stages.empty())
        stage_  = stages[0];
    layout_ = prog.pipelineLayout();
    return *this;
}

VkPipeline ComputePipelineBuilder::build(VkPipelineCache cache)
{
    VkComputePipelineCreateInfo pipeCI{};
    pipeCI.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeCI.stage  = stage_;
    pipeCI.layout = layout_;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(dev_->device(), cache, 1, &pipeCI, nullptr, &pipeline));
    return pipeline;
}

// ─── PipelineCache ───────────────────────────────────────────────────────────

void PipelineCache::init(RHIDevice& dev, const std::string& cacheFile)
{
    dev_       = &dev;
    cacheFile_ = cacheFile;

    VkPipelineCacheCreateInfo cacheCI{};
    cacheCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

    // 若缓存文件存在，从文件加载已序列化的缓存数据
    std::vector<char> cacheData;
    if (!cacheFile.empty()) {
        std::ifstream file(cacheFile, std::ios::ate | std::ios::binary);
        if (file.is_open()) {
            const size_t fileSize = static_cast<size_t>(file.tellg());
            cacheData.resize(fileSize);
            file.seekg(0);
            file.read(cacheData.data(), static_cast<std::streamsize>(fileSize));
            cacheCI.initialDataSize = cacheData.size();
            cacheCI.pInitialData    = cacheData.data();
        }
    }

    VK_CHECK(vkCreatePipelineCache(dev.device(), &cacheCI, nullptr, &vkCache_));
}

void PipelineCache::destroy()
{
    if (!dev_) return;
    for (auto& [hash, pipe] : runtimeCache_)
        if (pipe != VK_NULL_HANDLE)
            vkDestroyPipeline(dev_->device(), pipe, nullptr);
    runtimeCache_.clear();
    if (vkCache_ != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(dev_->device(), vkCache_, nullptr);
        vkCache_ = VK_NULL_HANDLE;
    }
    dev_ = nullptr;
}

void PipelineCache::save()
{
    if (!dev_ || vkCache_ == VK_NULL_HANDLE || cacheFile_.empty()) return;

    size_t dataSize = 0;
    vkGetPipelineCacheData(dev_->device(), vkCache_, &dataSize, nullptr);
    std::vector<char> data(dataSize);
    vkGetPipelineCacheData(dev_->device(), vkCache_, &dataSize, data.data());

    std::ofstream file(cacheFile_, std::ios::binary | std::ios::trunc);
    if (file.is_open())
        file.write(data.data(), static_cast<std::streamsize>(dataSize));
}

VkPipeline PipelineCache::find(size_t stateHash) const
{
    auto it = runtimeCache_.find(stateHash);
    return (it != runtimeCache_.end()) ? it->second : VK_NULL_HANDLE;
}

void PipelineCache::store(size_t stateHash, VkPipeline pipeline)
{
    runtimeCache_[stateHash] = pipeline;
}

} // namespace engine
