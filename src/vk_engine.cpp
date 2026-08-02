#include "vk_engine.h"
#include "vk_context.h"
#include "vk_window.h"

#include <array>
#include <memory>
#include <span>
#include <utility>

namespace vk_engine
{
VkEngine::VkEngine()
{
    window = std::make_unique<VkWindow>(800, 600);
    context = std::make_unique<VkContext>(*window.get());
    swapchain = std::make_unique<VkSwapchain>(*context, *window);

    const std::array<Vertex, 3> vertices{Vertex{glm::vec2{0.0F, -0.5F}, glm::vec3{1.0F, 0.0F, 0.0F}},
                                         Vertex{glm::vec2{0.5F, 0.5F}, glm::vec3{0.0F, 1.0F, 0.0F}},
                                         Vertex{glm::vec2{-0.5F, 0.5F}, glm::vec3{0.0F, 0.0F, 1.0F}}};
    triangleVertexBuffer = std::make_unique<VkBuffer>(*context,
                                                      sizeof(vertices),
                                                      vk::BufferUsageFlagBits::eVertexBuffer,
                                                      vk::MemoryPropertyFlagBits::eHostVisible |
                                                          vk::MemoryPropertyFlagBits::eHostCoherent);
    triangleVertexBuffer->Write(std::as_bytes(std::span<const Vertex>{vertices}));

    GraphicsPipelineDescription pipelineDescription{};
    pipelineDescription.vertexInput = Vertex::GetInputDescription();
    graphicsPipeline =
        std::make_unique<GraphicsPipeline>(*context, std::move(pipelineDescription), swapchain->GetImageFormat());
    renderer = std::make_unique<VkRenderer>(*context, *swapchain);
}

VkEngine::~VkEngine()
{
}

void VkEngine::Run()
{
    while (!window->ShouldClose())
    {
        window->ProcessPendingEvents();
        renderer->DrawFrame(
            [this](vk::CommandBuffer commandBuffer, const RenderTarget& target)
            {
                graphicsPipeline->EnsureCompatible(target.colorFormat);

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

                graphicsPipeline->Bind(commandBuffer);
                const vk::Viewport viewport{0.0F,
                                            0.0F,
                                            static_cast<float>(target.extent.width),
                                            static_cast<float>(target.extent.height),
                                            0.0F,
                                            1.0F};
                const vk::Rect2D scissor{{0, 0}, target.extent};
                commandBuffer.setViewport(0, viewport);
                commandBuffer.setScissor(0, scissor);

                const std::array<vk::Buffer, 1> buffers{triangleVertexBuffer->GetHandle()};
                const std::array<vk::DeviceSize, 1> offsets{0};
                commandBuffer.bindVertexBuffers(0, buffers, offsets);
                commandBuffer.draw(3, 1, 0, 0);

                commandBuffer.endRendering();
            });
    }
}
} // namespace vk_engine
