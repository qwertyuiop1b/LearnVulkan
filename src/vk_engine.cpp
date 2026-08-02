#include "vk_engine.h"

namespace vk_engine
{
VkEngine::VkEngine(uint32_t width, uint32_t height)
{
    window = std::make_unique<VkWindow>(width, height);
    context = std::make_unique<VkContext>(*window.get());
    swapchain = std::make_unique<VkSwapchain>(*context, *window);
    renderer = std::make_unique<VkRenderer>(*context, *swapchain);
}

VkEngine::~VkEngine()
{
}

void VkEngine::Run(const VkRenderer::RecordCallback& record)
{
    while (!window->ShouldClose())
    {
        window->ProcessPendingEvents();
        renderer->DrawFrame(record);
    }

    renderer->WaitIdle();
}

void VkEngine::WaitIdle() const
{
    renderer->WaitIdle();
}
} // namespace vk_engine
