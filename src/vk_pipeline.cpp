#include "vk_pipeline.h"

#include "vk_shader.h"

#include <array>

namespace vk_engine
{
GraphicsPipeline::GraphicsPipeline(const VkContext& inContext,
                                   GraphicsPipelineDescription inDescription,
                                   vk::Format initialColorFormat)
    : context(inContext), description(std::move(inDescription))
{
    vk::PipelineLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.setSetLayouts(description.pipelineLayout.descriptorSetLayouts)
        .setPushConstantRanges(description.pipelineLayout.pushConstantRanges);
    pipelineLayout = vk::raii::PipelineLayout(context.GetDevice(), layoutCreateInfo);

    CreatePipeline(initialColorFormat);
}

void GraphicsPipeline::EnsureCompatible(vk::Format newColorFormat)
{
    if (newColorFormat == colorFormat)
    {
        return;
    }

    CreatePipeline(newColorFormat);
}

void GraphicsPipeline::Bind(vk::CommandBuffer commandBuffer) const
{
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
}

void GraphicsPipeline::CreatePipeline(vk::Format newColorFormat)
{
    const ShaderModule vertexShader(context.GetDevice(), description.vertexShader);
    const ShaderModule fragmentShader(context.GetDevice(), description.fragmentShader);

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages{};
    shaderStages[0].setStage(vk::ShaderStageFlagBits::eVertex).setModule(vertexShader.GetHandle()).setPName("main");
    shaderStages[1].setStage(vk::ShaderStageFlagBits::eFragment).setModule(fragmentShader.GetHandle()).setPName("main");

    vk::PipelineVertexInputStateCreateInfo vertexInputState{};
    vertexInputState.setVertexBindingDescriptions(description.vertexInput.bindings)
        .setVertexAttributeDescriptions(description.vertexInput.attributes);

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState{};
    inputAssemblyState.setTopology(description.topology).setPrimitiveRestartEnable(false);

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.setViewportCount(1).setScissorCount(1);

    vk::PipelineRasterizationStateCreateInfo rasterizationState{};
    rasterizationState.setDepthClampEnable(false)
        .setRasterizerDiscardEnable(false)
        .setPolygonMode(description.polygonMode)
        .setCullMode(description.cullMode)
        .setFrontFace(description.frontFace)
        .setDepthBiasEnable(false)
        .setLineWidth(1.0F);

    vk::PipelineMultisampleStateCreateInfo multisampleState{};
    multisampleState.setRasterizationSamples(description.samples).setSampleShadingEnable(false);

    vk::PipelineDepthStencilStateCreateInfo depthStencilState{};
    depthStencilState.setDepthTestEnable(description.depthTestEnable)
        .setDepthWriteEnable(description.depthWriteEnable)
        .setDepthCompareOp(vk::CompareOp::eLess)
        .setDepthBoundsTestEnable(false)
        .setStencilTestEnable(false);

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.setBlendEnable(description.blendEnable)
        .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
        .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
        .setColorBlendOp(vk::BlendOp::eAdd)
        .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
        .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
        .setAlphaBlendOp(vk::BlendOp::eAdd)
        .setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                           vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

    vk::PipelineColorBlendStateCreateInfo colorBlendState{};
    colorBlendState.setLogicOpEnable(false).setAttachments(colorBlendAttachment);

    const std::array<vk::DynamicState, 2> dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.setDynamicStates(dynamicStates);

    vk::PipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.setColorAttachmentFormats(newColorFormat);

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.setStages(shaderStages)
        .setPVertexInputState(&vertexInputState)
        .setPInputAssemblyState(&inputAssemblyState)
        .setPViewportState(&viewportState)
        .setPRasterizationState(&rasterizationState)
        .setPMultisampleState(&multisampleState)
        .setPDepthStencilState(&depthStencilState)
        .setPColorBlendState(&colorBlendState)
        .setPDynamicState(&dynamicState)
        .setLayout(*pipelineLayout)
        .setRenderPass(nullptr)
        .setSubpass(0)
        .setPNext(&renderingInfo);

    vk::raii::Pipeline newPipeline(context.GetDevice(), nullptr, pipelineCreateInfo);
    pipeline = std::move(newPipeline);
    colorFormat = newColorFormat;
}
} // namespace vk_engine
