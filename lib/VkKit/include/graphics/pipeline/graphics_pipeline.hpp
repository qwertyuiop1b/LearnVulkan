#pragma once

#include <graphics/shader/pipeline_layout.hpp>
#include <graphics/shader/shader_module.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace vulkan_graphics {

struct GraphicsShaderStage {
    const ShaderModule* module = nullptr;
    vk::ShaderStageFlagBits stage = vk::ShaderStageFlagBits::eVertex;
    std::string entryPoint = "main";
};

struct VertexBindingDescription {
    uint32_t binding = 0;
    uint32_t stride = 0;
    vk::VertexInputRate inputRate = vk::VertexInputRate::eVertex;
};

struct VertexAttributeDescription {
    uint32_t location = 0;
    uint32_t binding = 0;
    vk::Format format = vk::Format::eUndefined;
    uint32_t offset = 0;
};

struct ColorBlendAttachmentState {
    bool blendEnable = false;
    vk::BlendFactor sourceColorBlendFactor = vk::BlendFactor::eOne;
    vk::BlendFactor destinationColorBlendFactor = vk::BlendFactor::eZero;
    vk::BlendOp colorBlendOp = vk::BlendOp::eAdd;
    vk::BlendFactor sourceAlphaBlendFactor = vk::BlendFactor::eOne;
    vk::BlendFactor destinationAlphaBlendFactor = vk::BlendFactor::eZero;
    vk::BlendOp alphaBlendOp = vk::BlendOp::eAdd;
    vk::ColorComponentFlags colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                             vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
};

struct StencilFaceState {
    vk::StencilOp failOp = vk::StencilOp::eKeep;
    vk::StencilOp passOp = vk::StencilOp::eKeep;
    vk::StencilOp depthFailOp = vk::StencilOp::eKeep;
    vk::CompareOp compareOp = vk::CompareOp::eAlways;
    uint32_t compareMask = 0;
    uint32_t writeMask = 0;
    uint32_t reference = 0;
};

struct GraphicsPipelineCreateInfo {
    const PipelineLayout* layout = nullptr;
    std::vector<GraphicsShaderStage> shaderStages;
    std::vector<VertexBindingDescription> vertexBindings;
    std::vector<VertexAttributeDescription> vertexAttributes;
    std::vector<vk::Format> colorAttachmentFormats;
    std::vector<ColorBlendAttachmentState> colorBlendAttachments;
    vk::Format depthAttachmentFormat = vk::Format::eUndefined;
    vk::Format stencilAttachmentFormat = vk::Format::eUndefined;
    vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList;
    bool primitiveRestartEnable = false;
    vk::PolygonMode polygonMode = vk::PolygonMode::eFill;
    vk::CullModeFlags cullMode = vk::CullModeFlagBits::eBack;
    vk::FrontFace frontFace = vk::FrontFace::eCounterClockwise;
    float lineWidth = 1.0f;
    vk::SampleCountFlagBits rasterizationSamples = vk::SampleCountFlagBits::e1;
    bool depthTestEnable = false;
    bool depthWriteEnable = false;
    vk::CompareOp depthCompareOp = vk::CompareOp::eLess;
    bool stencilTestEnable = false;
    StencilFaceState frontStencil{};
    StencilFaceState backStencil{};
};

class GraphicsPipeline final {
  public:
    GraphicsPipeline() = default;
    GraphicsPipeline(const VulkanContext& context, const GraphicsPipelineCreateInfo& createInfo);
    ~GraphicsPipeline() noexcept;

    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(GraphicsPipeline&& other) noexcept;
    GraphicsPipeline& operator=(GraphicsPipeline&& other) noexcept;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] vk::Pipeline handle() const noexcept;
    [[nodiscard]] VkPipeline nativeHandle() const noexcept;
    [[nodiscard]] const PipelineLayout* layout() const noexcept;

  private:
    void destroy() noexcept;

    const VulkanContext* context_ = nullptr;
    const PipelineLayout* layout_ = nullptr;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

} // namespace vulkan_graphics
