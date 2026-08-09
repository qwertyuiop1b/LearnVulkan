#include "triangle_vertex.h"

#include "vk_buffer.h"
#include "vk_engine.h"
#include "vk_pipeline.h"
#include "vk_shader.h"

#include <array>
#include <span>
#include <utility>

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
    pipelineDescription.vertexShader = vk_engine::ShaderPath("simple.vert.spv");
    pipelineDescription.fragmentShader = vk_engine::ShaderPath("simple.frag.spv");
    pipelineDescription.vertexInput = triangle_example::Vertex::GetInputDescription();
    vk_engine::GraphicsPipeline graphicsPipeline(
        engine.GetContext(), std::move(pipelineDescription), engine.GetSwapchain().GetImageFormat());
    engine.Run(
        [&](vk::CommandBuffer commandBuffer, vk_engine::RenderHelper& helper)
        {
            helper.TransitionToGraphics();
            const vk::Format colorFormat = helper.GetDrawImageFormat();
            graphicsPipeline.EnsureCompatible(colorFormat);
            const vk::Extent2D extent = helper.GetDrawExtent();
            vk::ClearValue clearValue{};
            clearValue.setColor(vk::ClearColorValue{std::array<float, 4>{0.05F, 0.1F, 0.2F, 1.0F}});
            vk::RenderingAttachmentInfo colorAttachment{};
            colorAttachment.setImageView(helper.GetDrawImageView())
                .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setClearValue(clearValue);
            vk::RenderingInfo renderingInfo{};
            renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, extent})
                .setLayerCount(1)
                .setColorAttachments(colorAttachment);
            commandBuffer.beginRendering(renderingInfo);
            graphicsPipeline.Bind(commandBuffer);
            const vk::Viewport viewport{
                0.0F, 0.0F, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0F, 1.0F};
            const vk::Rect2D scissor{{0, 0}, extent};
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
