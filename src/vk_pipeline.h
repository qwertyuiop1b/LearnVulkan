#pragma once

#include "vk_context.h"

#include <filesystem>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

namespace vk_engine
{
struct VertexInputDescription
{
    std::vector<vk::VertexInputBindingDescription> bindings;
    std::vector<vk::VertexInputAttributeDescription> attributes;
};

struct PipelineLayoutDescription
{
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
    std::vector<vk::PushConstantRange> pushConstantRanges;
};

struct GraphicsPipelineDescription
{
    std::filesystem::path vertexShader;
    std::filesystem::path fragmentShader;
    VertexInputDescription vertexInput;
    PipelineLayoutDescription pipelineLayout;
    vk::PrimitiveTopology topology{vk::PrimitiveTopology::eTriangleList};
    vk::PolygonMode polygonMode{vk::PolygonMode::eFill};
    vk::CullModeFlags cullMode{};
    vk::FrontFace frontFace{vk::FrontFace::eCounterClockwise};
    vk::SampleCountFlagBits samples{vk::SampleCountFlagBits::e1};
    bool depthTestEnable{false};
    bool depthWriteEnable{false};
    bool blendEnable{false};
};

class GraphicsPipeline
{
public:
    GraphicsPipeline(const VkContext& context, GraphicsPipelineDescription description, vk::Format initialColorFormat);
    ~GraphicsPipeline() = default;

    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(GraphicsPipeline&&) = delete;
    GraphicsPipeline& operator=(GraphicsPipeline&&) = delete;

    void EnsureCompatible(vk::Format colorFormat);
    void Bind(vk::CommandBuffer commandBuffer) const;

    vk::PipelineLayout GetLayout() const noexcept
    {
        return *pipelineLayout;
    }

private:
    void CreatePipeline(vk::Format colorFormat);

    const VkContext& context;
    GraphicsPipelineDescription description;
    vk::raii::PipelineLayout pipelineLayout{nullptr};
    vk::raii::Pipeline pipeline{nullptr};
    vk::Format colorFormat{vk::Format::eUndefined};
};
} // namespace vk_engine
