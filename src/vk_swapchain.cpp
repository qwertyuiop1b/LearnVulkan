#include "vk_swapchain.h"
#include <stdexcept>
#include <VkBootstrap.h>

namespace vk_engine
{
VkSwapchain::VkSwapchain(const VkContext& inContext, const VkWindow& inWindow) : context(inContext), window(inWindow)
{
    Create(VK_NULL_HANDLE);
}

void VkSwapchain::Recreate()
{
    const VkSwapchainKHR oldSwapchain = static_cast<VkSwapchainKHR>(static_cast<vk::SwapchainKHR>(swapchain));
    Create(oldSwapchain);
}

bool VkSwapchain::HasValidFramebufferExtent() const noexcept
{
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(&window.GetWindow(), &framebufferWidth, &framebufferHeight);
    return framebufferWidth > 0 && framebufferHeight > 0;
}

void VkSwapchain::Create(VkSwapchainKHR oldSwapchain)
{
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(&window.GetWindow(), &framebufferWidth, &framebufferHeight);
    if (framebufferWidth <= 0 || framebufferHeight <= 0)
    {
        throw std::runtime_error("cannot create a swapchain for a zero-sized framebuffer");
    }

    vkb::SwapchainBuilder builder{context.GetPhysicalDeviceHandle(),
                                  context.GetDeviceHandle(),
                                  context.GetSurfaceHandle(),
                                  context.GetGraphicQueueFamilyIndex(),
                                  context.GetPresentQueueFamilyIndex()};

    builder.set_desired_extent(static_cast<uint32_t>(framebufferWidth), static_cast<uint32_t>(framebufferHeight))
        .use_default_format_selection()
        .use_default_present_mode_selection()
        .set_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

    if (oldSwapchain != VK_NULL_HANDLE)
    {
        builder.set_old_swapchain(oldSwapchain);
    }

    const auto vkbSwapchainResult = builder.build();
    if (!vkbSwapchainResult)
    {
        if (oldSwapchain != VK_NULL_HANDLE &&
            (vkbSwapchainResult.matches_error(vkb::SwapchainError::failed_create_swapchain) ||
             vkbSwapchainResult.matches_error(vkb::SwapchainError::failed_get_swapchain_images)))
        {
            // vk-bootstrap documents the old handle as invalid after a failed rebuild.
            swapchainImageViews.clear();
            swapchainImages.clear();
            (void)swapchain.release();
        }

        throw std::runtime_error("failed to create swapchain: " + vkbSwapchainResult.error().message());
    }

    vkb::Swapchain vkbSwapchain = vkbSwapchainResult.value();
    vk::raii::SwapchainKHR newSwapchain(context.GetDevice(), vkbSwapchain.swapchain);
    vkbSwapchain.swapchain = VK_NULL_HANDLE;

    std::vector<vk::Image> newImages = newSwapchain.getImages();
    std::vector<vk::raii::ImageView> newImageViews;
    newImageViews.reserve(newImages.size());

    const vk::Format newFormat = static_cast<vk::Format>(vkbSwapchain.image_format);
    for (const vk::Image image : newImages)
    {
        const vk::ImageViewCreateInfo imageViewCreateInfo{{},
                                                          image,
                                                          vk::ImageViewType::e2D,
                                                          newFormat,
                                                          {vk::ComponentSwizzle::eIdentity,
                                                           vk::ComponentSwizzle::eIdentity,
                                                           vk::ComponentSwizzle::eIdentity,
                                                           vk::ComponentSwizzle::eIdentity},
                                                          {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        newImageViews.emplace_back(context.GetDevice(), imageViewCreateInfo);
    }

    swapchainImageViews = std::move(newImageViews);
    swapchainImages = std::move(newImages);
    format = newFormat;
    extent = vk::Extent2D{vkbSwapchain.extent.width, vkbSwapchain.extent.height};
    swapchain = std::move(newSwapchain);
}
} // namespace vk_engine
