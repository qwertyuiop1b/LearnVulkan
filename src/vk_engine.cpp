#include "vk_engine.h"
#include "vk_context.h"
#include "vk_window.h"

#include <array>
#include <memory>

namespace vk_engine
{
VkEngine::VkEngine()
{
    window = std::make_unique<VkWindow>(800, 600);
    context = std::make_unique<VkContext>(*window.get());
    swapchain = std::make_unique<VkSwapchain>(*context, *window);
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
            [](vk::CommandBuffer commandBuffer, vk::Image, vk::ImageView imageView, vk::Extent2D extent)
            {
                vk::ClearValue clearValue{};
                clearValue.setColor(vk::ClearColorValue{std::array<float, 4>{0.05F, 0.1F, 0.2F, 1.0F}});

                vk::RenderingAttachmentInfo colorAttachment{};
                colorAttachment.setImageView(imageView)
                    .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                    .setLoadOp(vk::AttachmentLoadOp::eClear)
                    .setStoreOp(vk::AttachmentStoreOp::eStore)
                    .setClearValue(clearValue);

                vk::RenderingInfo renderingInfo{};
                renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, extent})
                    .setLayerCount(1)
                    .setColorAttachments(colorAttachment);

                commandBuffer.beginRendering(renderingInfo);
                commandBuffer.endRendering();
            });
    }
}
} // namespace vk_engine
