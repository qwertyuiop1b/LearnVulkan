#pragma once

#include "vk_context.h"
#include "vk_renderer.h"
#include "vk_swapchain.h"
#include "vk_window.h"

#include <cstdint>
#include <memory>

namespace vk_engine
{
class VkEngine
{
public:
    explicit VkEngine(uint32_t width = 800, uint32_t height = 600);
    ~VkEngine();

    void Run(const VkRenderer::RenderCallback& callback);
    void WaitIdle() const;

    const VkContext& GetContext() const noexcept
    {
        return *context;
    }

    const VkSwapchain& GetSwapchain() const noexcept
    {
        return *swapchain;
    }

private:
    std::unique_ptr<VkWindow> window{nullptr};
    std::unique_ptr<VkContext> context{nullptr};
    std::unique_ptr<VkSwapchain> swapchain{nullptr};
    std::unique_ptr<VkRenderer> renderer{nullptr};
};
} // namespace vk_engine
