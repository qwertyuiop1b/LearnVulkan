#include <graphics/graphics_pipeline.hpp>

#include <set>
#include <stdexcept>
#include <utility>

namespace vulkan_graphics {
namespace {

VkStencilOpState toNativeStencilState(const StencilFaceState& state) {
    return {
        static_cast<VkStencilOp>(state.failOp),
        static_cast<VkStencilOp>(state.passOp),
        static_cast<VkStencilOp>(state.depthFailOp),
        static_cast<VkCompareOp>(state.compareOp),
        state.compareMask,
        state.writeMask,
        state.reference,
    };
}

VkPipelineColorBlendAttachmentState toNativeBlendState(const ColorBlendAttachmentState& state) {
    return {
        state.blendEnable,
        static_cast<VkBlendFactor>(state.sourceColorBlendFactor),
        static_cast<VkBlendFactor>(state.destinationColorBlendFactor),
        static_cast<VkBlendOp>(state.colorBlendOp),
        static_cast<VkBlendFactor>(state.sourceAlphaBlendFactor),
        static_cast<VkBlendFactor>(state.destinationAlphaBlendFactor),
        static_cast<VkBlendOp>(state.alphaBlendOp),
        static_cast<VkColorComponentFlags>(state.colorWriteMask),
    };
}

} // namespace

GraphicsPipeline::GraphicsPipeline(const VulkanContext& context, const GraphicsPipelineCreateInfo& createInfo)
    : context_(&context), layout_(createInfo.layout) {
    if (static_cast<VkDevice>(context.device()) == VK_NULL_HANDLE)
        throw std::invalid_argument("GraphicsPipeline requires an initialized VulkanContext");
    if (!context.dynamicRenderingEnabled())
        throw std::invalid_argument("GraphicsPipeline requires dynamic rendering enabled in VulkanContext");
    if (layout_ == nullptr || !layout_->isValid())
        throw std::invalid_argument("GraphicsPipeline requires a valid PipelineLayout");
    if (createInfo.shaderStages.empty())
        throw std::invalid_argument("GraphicsPipeline requires at least a vertex shader stage");
    if (createInfo.polygonMode != vk::PolygonMode::eFill)
        throw std::invalid_argument("GraphicsPipeline currently supports fill polygon mode only");
    if (createInfo.lineWidth != 1.0f)
        throw std::invalid_argument("GraphicsPipeline currently supports line width of one only");
    if ((createInfo.depthTestEnable || createInfo.depthWriteEnable) &&
        createInfo.depthAttachmentFormat == vk::Format::eUndefined)
        throw std::invalid_argument("Depth testing requires a depth attachment format");
    if (createInfo.stencilTestEnable && createInfo.stencilAttachmentFormat == vk::Format::eUndefined)
        throw std::invalid_argument("Stencil testing requires a stencil attachment format");
    if (!createInfo.colorBlendAttachments.empty() &&
        createInfo.colorBlendAttachments.size() != createInfo.colorAttachmentFormats.size())
        throw std::invalid_argument("Color blend attachment count must match color attachment formats");

    std::set<vk::ShaderStageFlagBits> shaderStages;
    bool hasVertexShader = false;
    std::vector<VkPipelineShaderStageCreateInfo> nativeShaderStages;
    nativeShaderStages.reserve(createInfo.shaderStages.size());
    for (const GraphicsShaderStage& shaderStage : createInfo.shaderStages) {
        if (shaderStage.module == nullptr || !shaderStage.module->isValid())
            throw std::invalid_argument("GraphicsPipeline requires valid shader modules");
        if (shaderStage.entryPoint.empty())
            throw std::invalid_argument("GraphicsPipeline shader entry points must not be empty");
        if (shaderStage.stage != vk::ShaderStageFlagBits::eVertex && shaderStage.stage != vk::ShaderStageFlagBits::eFragment)
            throw std::invalid_argument("GraphicsPipeline currently supports vertex and fragment shader stages only");
        if (!shaderStages.insert(shaderStage.stage).second)
            throw std::invalid_argument("GraphicsPipeline shader stages must be unique");
        hasVertexShader = hasVertexShader || shaderStage.stage == vk::ShaderStageFlagBits::eVertex;

        VkPipelineShaderStageCreateInfo nativeShaderStage{};
        nativeShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        nativeShaderStage.stage = static_cast<VkShaderStageFlagBits>(shaderStage.stage);
        nativeShaderStage.module = shaderStage.module->nativeHandle();
        nativeShaderStage.pName = shaderStage.entryPoint.c_str();
        nativeShaderStages.push_back(nativeShaderStage);
    }
    if (!hasVertexShader)
        throw std::invalid_argument("GraphicsPipeline requires a vertex shader stage");

    std::set<uint32_t> vertexBindings;
    std::vector<VkVertexInputBindingDescription> nativeVertexBindings;
    nativeVertexBindings.reserve(createInfo.vertexBindings.size());
    for (const VertexBindingDescription& binding : createInfo.vertexBindings) {
        if (binding.stride == 0)
            throw std::invalid_argument("Vertex binding stride must be greater than zero");
        if (!vertexBindings.insert(binding.binding).second)
            throw std::invalid_argument("Vertex binding indices must be unique");
        nativeVertexBindings.push_back(
            {binding.binding, binding.stride, static_cast<VkVertexInputRate>(binding.inputRate)});
    }

    std::set<uint32_t> vertexLocations;
    std::vector<VkVertexInputAttributeDescription> nativeVertexAttributes;
    nativeVertexAttributes.reserve(createInfo.vertexAttributes.size());
    for (const VertexAttributeDescription& attribute : createInfo.vertexAttributes) {
        if (attribute.format == vk::Format::eUndefined)
            throw std::invalid_argument("Vertex attribute formats must be specified");
        if (!vertexBindings.count(attribute.binding))
            throw std::invalid_argument("Vertex attributes must reference an existing binding");
        if (!vertexLocations.insert(attribute.location).second)
            throw std::invalid_argument("Vertex attribute locations must be unique");
        nativeVertexAttributes.push_back(
            {attribute.location, attribute.binding, static_cast<VkFormat>(attribute.format), attribute.offset});
    }

    std::vector<VkFormat> nativeColorFormats;
    nativeColorFormats.reserve(createInfo.colorAttachmentFormats.size());
    for (const vk::Format format : createInfo.colorAttachmentFormats) {
        if (format == vk::Format::eUndefined)
            throw std::invalid_argument("Color attachment formats must be specified");
        nativeColorFormats.push_back(static_cast<VkFormat>(format));
    }

    std::vector<ColorBlendAttachmentState> blendAttachments = createInfo.colorBlendAttachments;
    if (blendAttachments.empty())
        blendAttachments.resize(nativeColorFormats.size());
    std::vector<VkPipelineColorBlendAttachmentState> nativeBlendAttachments;
    nativeBlendAttachments.reserve(blendAttachments.size());
    for (const ColorBlendAttachmentState& attachment : blendAttachments)
        nativeBlendAttachments.push_back(toNativeBlendState(attachment));

    VkPipelineVertexInputStateCreateInfo vertexInputState{};
    vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(nativeVertexBindings.size());
    vertexInputState.pVertexBindingDescriptions = nativeVertexBindings.data();
    vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(nativeVertexAttributes.size());
    vertexInputState.pVertexAttributeDescriptions = nativeVertexAttributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{};
    inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyState.topology = static_cast<VkPrimitiveTopology>(createInfo.topology);
    inputAssemblyState.primitiveRestartEnable = createInfo.primitiveRestartEnable;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizationState{};
    rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationState.depthClampEnable = VK_FALSE;
    rasterizationState.rasterizerDiscardEnable = VK_FALSE;
    rasterizationState.polygonMode = static_cast<VkPolygonMode>(createInfo.polygonMode);
    rasterizationState.cullMode = static_cast<VkCullModeFlags>(createInfo.cullMode);
    rasterizationState.frontFace = static_cast<VkFrontFace>(createInfo.frontFace);
    rasterizationState.depthBiasEnable = VK_FALSE;
    rasterizationState.lineWidth = createInfo.lineWidth;

    VkPipelineMultisampleStateCreateInfo multisampleState{};
    multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleState.rasterizationSamples = static_cast<VkSampleCountFlagBits>(createInfo.rasterizationSamples);
    multisampleState.sampleShadingEnable = VK_FALSE;
    multisampleState.alphaToCoverageEnable = VK_FALSE;
    multisampleState.alphaToOneEnable = VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo depthStencilState{};
    depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilState.depthTestEnable = createInfo.depthTestEnable;
    depthStencilState.depthWriteEnable = createInfo.depthWriteEnable;
    depthStencilState.depthCompareOp = static_cast<VkCompareOp>(createInfo.depthCompareOp);
    depthStencilState.depthBoundsTestEnable = VK_FALSE;
    depthStencilState.stencilTestEnable = createInfo.stencilTestEnable;
    depthStencilState.front = toNativeStencilState(createInfo.frontStencil);
    depthStencilState.back = toNativeStencilState(createInfo.backStencil);

    VkPipelineColorBlendStateCreateInfo colorBlendState{};
    colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendState.logicOpEnable = VK_FALSE;
    colorBlendState.attachmentCount = static_cast<uint32_t>(nativeBlendAttachments.size());
    colorBlendState.pAttachments = nativeBlendAttachments.data();

    constexpr VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(std::size(dynamicStates));
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = static_cast<uint32_t>(nativeColorFormats.size());
    renderingInfo.pColorAttachmentFormats = nativeColorFormats.data();
    renderingInfo.depthAttachmentFormat = static_cast<VkFormat>(createInfo.depthAttachmentFormat);
    renderingInfo.stencilAttachmentFormat = static_cast<VkFormat>(createInfo.stencilAttachmentFormat);

    VkGraphicsPipelineCreateInfo nativeCreateInfo{};
    nativeCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    nativeCreateInfo.pNext = &renderingInfo;
    nativeCreateInfo.stageCount = static_cast<uint32_t>(nativeShaderStages.size());
    nativeCreateInfo.pStages = nativeShaderStages.data();
    nativeCreateInfo.pVertexInputState = &vertexInputState;
    nativeCreateInfo.pInputAssemblyState = &inputAssemblyState;
    nativeCreateInfo.pViewportState = &viewportState;
    nativeCreateInfo.pRasterizationState = &rasterizationState;
    nativeCreateInfo.pMultisampleState = &multisampleState;
    nativeCreateInfo.pDepthStencilState = &depthStencilState;
    nativeCreateInfo.pColorBlendState = &colorBlendState;
    nativeCreateInfo.pDynamicState = &dynamicState;
    nativeCreateInfo.layout = layout_->nativeHandle();
    nativeCreateInfo.renderPass = VK_NULL_HANDLE;
    nativeCreateInfo.subpass = 0;
    nativeCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    nativeCreateInfo.basePipelineIndex = -1;
    if (vkCreateGraphicsPipelines(static_cast<VkDevice>(context.device()), VK_NULL_HANDLE, 1, &nativeCreateInfo,
                                  nullptr, &pipeline_) != VK_SUCCESS) {
        context_ = nullptr;
        layout_ = nullptr;
        throw std::runtime_error("Failed to create Vulkan graphics pipeline");
    }
}

GraphicsPipeline::~GraphicsPipeline() noexcept {
    destroy();
}

GraphicsPipeline::GraphicsPipeline(GraphicsPipeline&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      layout_(std::exchange(other.layout_, nullptr)),
      pipeline_(std::exchange(other.pipeline_, VK_NULL_HANDLE)) {}

GraphicsPipeline& GraphicsPipeline::operator=(GraphicsPipeline&& other) noexcept {
    if (this == &other)
        return *this;

    destroy();
    context_ = std::exchange(other.context_, nullptr);
    layout_ = std::exchange(other.layout_, nullptr);
    pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
    return *this;
}

bool GraphicsPipeline::isValid() const noexcept {
    return pipeline_ != VK_NULL_HANDLE;
}

vk::Pipeline GraphicsPipeline::handle() const noexcept {
    return vk::Pipeline{pipeline_};
}

VkPipeline GraphicsPipeline::nativeHandle() const noexcept {
    return pipeline_;
}

const PipelineLayout* GraphicsPipeline::layout() const noexcept {
    return layout_;
}

void GraphicsPipeline::destroy() noexcept {
    if (pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(static_cast<VkDevice>(context_->device()), pipeline_, nullptr);

    context_ = nullptr;
    layout_ = nullptr;
    pipeline_ = VK_NULL_HANDLE;
}

} // namespace vulkan_graphics
