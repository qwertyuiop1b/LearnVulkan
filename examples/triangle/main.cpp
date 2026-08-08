#include "triangle_vertex.h"

#include "vk_buffer.h"
#include "vk_engine.h"
#include "vk_pipeline.h"

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <utility>

namespace
{
std::filesystem::path ShaderPath(const char* filename)
{
#ifdef VK_ENGINE_TRIANGLE_SHADER_DIR
    return std::filesystem::path{VK_ENGINE_TRIANGLE_SHADER_DIR} / filename;
#else
    return std::filesystem::path{"shaders"} / filename;
#endif
}
} // namespace

int main()
{
    vk_engine::VkEngine engine{};

    const std::array<triangle_example::Vertex, 3> vertices{
        triangle_example::Vertex{glm::vec2{0.0F, -0.5F}, glm::vec3{1.0F, 0.0F, 0.0F}},
        triangle_example::Vertex{glm::vec2{0.5F, 0.5F}, glm::vec3{0.0F, 1.0F, 0.0F}},
        triangle_example::Vertex{glm::vec2{-0.5F, 0.5F}, glm::vec3{0.0F, 0.0F, 1.0F}}};
    vk_engine::Buffer vertexBuffer(engine.GetContext(),
                                     sizeof(vertices),
                                     vk::BufferUsageFlagBits::eVertexBuffer,
                                     vk::MemoryPropertyFlagBits::eHostVisible |
                                         vk::MemoryPropertyFlagBits::eHostCoherent);
    vertexBuffer.Write(std::as_bytes(std::span<const triangle_example::Vertex>{vertices}));

    vk_engine::GraphicsPipelineDescription pipelineDescription{};
    pipelineDescription.vertexShader = ShaderPath("simple.vert.spv");
    pipelineDescription.fragmentShader = ShaderPath("simple.frag.spv");
    pipelineDescription.vertexInput = triangle_example::Vertex::GetInputDescription();
    vk_engine::GraphicsPipeline graphicsPipeline(
        engine.GetContext(), std::move(pipelineDescription), engine.GetSwapchain().GetImageFormat());

    engine.Run(
        [&](vk::CommandBuffer commandBuffer, const vk_engine::RenderTarget& target)
        {
            graphicsPipeline.EnsureCompatible(target.colorFormat);

            vk::ClearValue clearValue{};
            clearValue.setColor(vk::ClearColorValue{std::array<float, 4>{0.05F, 0.1F, 0.2F, 1.0F}});

            vk::RenderingAttachmentInfo colorAttachment{};
            colorAttachment.setImageView(target.imageView)
                .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setClearValue(clearValue);

            vk::RenderingInfo renderingInfo{};
            renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, target.extent})
                .setLayerCount(1)
                .setColorAttachments(colorAttachment);

            commandBuffer.beginRendering(renderingInfo);
            graphicsPipeline.Bind(commandBuffer);

            const vk::Viewport viewport{0.0F,
                                        0.0F,
                                        static_cast<float>(target.extent.width),
                                        static_cast<float>(target.extent.height),
                                        0.0F,
                                        1.0F};
            const vk::Rect2D scissor{{0, 0}, target.extent};
            commandBuffer.setViewport(0, viewport);
            commandBuffer.setScissor(0, scissor);

            const std::array<vk::Buffer, 1> buffers{vertexBuffer.GetHandle()};
            const std::array<vk::DeviceSize, 1> offsets{0};
            commandBuffer.bindVertexBuffers(0, buffers, offsets);
            commandBuffer.draw(3, 1, 0, 0);
            commandBuffer.endRendering();
        });

    return 0;
}
