#include <graphics/render/swapchain.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vulkan_graphics {
namespace {

VkExtent2D chooseExtent(const vk::SurfaceCapabilitiesKHR& capabilities, VkExtent2D desiredExtent) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return static_cast<VkExtent2D>(capabilities.currentExtent);

    if (desiredExtent.width == 0 || desiredExtent.height == 0)
        throw std::invalid_argument("Swapchain requires a non-zero desired extent");

    return {
        std::clamp(desiredExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp(desiredExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
    };
}

vk::SurfaceFormatKHR chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats,
                                         const std::vector<vk::SurfaceFormatKHR>& preferredFormats) {
    for (const auto& preferred : preferredFormats) {
        for (const auto& available : availableFormats) {
            if (available.format == preferred.format && available.colorSpace == preferred.colorSpace)
                return available;
        }
    }

    return availableFormats.front();
}

vk::PresentModeKHR choosePresentMode(const std::vector<vk::PresentModeKHR>& availableModes,
                                     const std::vector<vk::PresentModeKHR>& preferredModes) {
    for (const auto preferred : preferredModes) {
        if (std::find(availableModes.begin(), availableModes.end(), preferred) != availableModes.end())
            return preferred;
    }

    return vk::PresentModeKHR::eFifo;
}

VkCompositeAlphaFlagBitsKHR chooseCompositeAlpha(const vk::SurfaceCapabilitiesKHR& capabilities) {
    constexpr VkCompositeAlphaFlagBitsKHR candidates[] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    };
    for (const VkCompositeAlphaFlagBitsKHR candidate : candidates) {
        if ((static_cast<VkCompositeAlphaFlagsKHR>(capabilities.supportedCompositeAlpha) & candidate) != 0)
            return candidate;
    }

    throw std::runtime_error("Surface does not expose a supported composite alpha mode");
}

SwapchainStatus toSwapchainStatus(VkResult result) {
    switch (result) {
    case VK_SUCCESS:
        return SwapchainStatus::eSuccess;
    case VK_SUBOPTIMAL_KHR:
        return SwapchainStatus::eSuboptimal;
    case VK_ERROR_OUT_OF_DATE_KHR:
        return SwapchainStatus::eOutOfDate;
    default:
        throw std::runtime_error("Vulkan swapchain operation failed");
    }
}

} // namespace

struct Swapchain::Resources {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    vk::Format format = vk::Format::eUndefined;
    vk::ColorSpaceKHR colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
    VkExtent2D extent{};
    std::vector<RenderTarget> renderTargets;
};

Swapchain::Swapchain(const VulkanContext& context, const SwapchainCreateInfo& createInfo) : context_(&context) {
    if (!context.device() || !context.surface())
        throw std::invalid_argument("Swapchain requires an initialized VulkanContext with a surface");

    createInfo_ = createInfo;
    assign(createResources(context, createInfo_, VK_NULL_HANDLE));
}

Swapchain::~Swapchain() noexcept {
    destroy();
}

Swapchain::Swapchain(Swapchain&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      createInfo_(std::move(other.createInfo_)),
      swapchain_(std::exchange(other.swapchain_, VK_NULL_HANDLE)),
      format_(std::exchange(other.format_, vk::Format::eUndefined)),
      colorSpace_(std::exchange(other.colorSpace_, vk::ColorSpaceKHR::eSrgbNonlinear)),
      presentMode_(std::exchange(other.presentMode_, vk::PresentModeKHR::eFifo)),
      extent_(std::exchange(other.extent_, {})),
      renderTargets_(std::move(other.renderTargets_)) {}

Swapchain& Swapchain::operator=(Swapchain&& other) noexcept {
    if (this == &other)
        return *this;

    destroy();
    context_ = std::exchange(other.context_, nullptr);
    createInfo_ = std::move(other.createInfo_);
    swapchain_ = std::exchange(other.swapchain_, VK_NULL_HANDLE);
    format_ = std::exchange(other.format_, vk::Format::eUndefined);
    colorSpace_ = std::exchange(other.colorSpace_, vk::ColorSpaceKHR::eSrgbNonlinear);
    presentMode_ = std::exchange(other.presentMode_, vk::PresentModeKHR::eFifo);
    extent_ = std::exchange(other.extent_, {});
    renderTargets_ = std::move(other.renderTargets_);
    return *this;
}

void Swapchain::recreate(VkExtent2D desiredExtent) {
    if (context_ == nullptr || swapchain_ == VK_NULL_HANDLE)
        throw std::logic_error("Cannot recreate an invalid Swapchain");

    createInfo_.desiredExtent = desiredExtent;
    context_->waitIdle();
    assign(createResources(*context_, createInfo_, swapchain_));
}

SwapchainAcquireResult Swapchain::acquireNextImage(VkSemaphore imageAvailableSemaphore, uint64_t timeout) const {
    if (swapchain_ == VK_NULL_HANDLE)
        throw std::logic_error("Cannot acquire an image from an invalid Swapchain");
    if (imageAvailableSemaphore == VK_NULL_HANDLE)
        throw std::invalid_argument("Swapchain acquisition requires an image-available semaphore");

    uint32_t imageIndex = 0;
    const VkResult result = vkAcquireNextImageKHR(static_cast<VkDevice>(context_->device()), swapchain_, timeout,
                                                   imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
    return {toSwapchainStatus(result), imageIndex};
}

SwapchainStatus Swapchain::present(uint32_t imageIndex, VkSemaphore renderFinishedSemaphore) const {
    if (swapchain_ == VK_NULL_HANDLE)
        throw std::logic_error("Cannot present an image from an invalid Swapchain");
    if (imageIndex >= renderTargets_.size())
        throw std::out_of_range("Swapchain image index is out of range");

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = renderFinishedSemaphore == VK_NULL_HANDLE ? 0U : 1U;
    presentInfo.pWaitSemaphores = renderFinishedSemaphore == VK_NULL_HANDLE ? nullptr : &renderFinishedSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_;
    presentInfo.pImageIndices = &imageIndex;
    return toSwapchainStatus(vkQueuePresentKHR(context_->presentQueue().handle, &presentInfo));
}

bool Swapchain::isValid() const noexcept {
    return swapchain_ != VK_NULL_HANDLE;
}

vk::SwapchainKHR Swapchain::handle() const noexcept {
    return vk::SwapchainKHR{swapchain_};
}

VkSwapchainKHR Swapchain::nativeHandle() const noexcept {
    return swapchain_;
}

vk::Format Swapchain::format() const noexcept {
    return format_;
}

vk::ColorSpaceKHR Swapchain::colorSpace() const noexcept {
    return colorSpace_;
}

vk::PresentModeKHR Swapchain::presentMode() const noexcept {
    return presentMode_;
}

VkExtent2D Swapchain::extent() const noexcept {
    return extent_;
}

uint32_t Swapchain::imageCount() const noexcept {
    return static_cast<uint32_t>(renderTargets_.size());
}

const RenderTarget& Swapchain::renderTarget(uint32_t imageIndex) const {
    if (imageIndex >= renderTargets_.size())
        throw std::out_of_range("Swapchain image index is out of range");

    return renderTargets_[imageIndex];
}

Swapchain::Resources Swapchain::createResources(const VulkanContext& context,
                                                 const SwapchainCreateInfo& createInfo,
                                                 VkSwapchainKHR oldSwapchain) {
    const auto capabilities = context.physicalDevice().getSurfaceCapabilitiesKHR(context.surface());
    const auto formats = context.physicalDevice().getSurfaceFormatsKHR(context.surface());
    const auto presentModes = context.physicalDevice().getSurfacePresentModesKHR(context.surface());
    if (formats.empty() || presentModes.empty())
        throw std::runtime_error("Surface does not support swapchain presentation");
    if ((capabilities.supportedUsageFlags & createInfo.imageUsage) != createInfo.imageUsage)
        throw std::invalid_argument("Requested swapchain image usage is not supported by the surface");

    const vk::SurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(formats, createInfo.preferredFormats);
    const vk::PresentModeKHR presentMode = choosePresentMode(presentModes, createInfo.preferredPresentModes);
    const VkExtent2D extent = chooseExtent(capabilities, createInfo.desiredExtent);
    uint32_t imageCount = createInfo.desiredImageCount;
    if (imageCount == 0)
        imageCount = capabilities.minImageCount + 1;
    imageCount = std::max(imageCount, capabilities.minImageCount);
    if (capabilities.maxImageCount != 0)
        imageCount = std::min(imageCount, capabilities.maxImageCount);

    const uint32_t queueFamilyIndices[] = {
        context.graphicsQueue().familyIndex,
        context.presentQueue().familyIndex,
    };
    const bool separatePresentQueue = queueFamilyIndices[0] != queueFamilyIndices[1];

    VkSwapchainCreateInfoKHR nativeCreateInfo{};
    nativeCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    nativeCreateInfo.surface = static_cast<VkSurfaceKHR>(context.surface());
    nativeCreateInfo.minImageCount = imageCount;
    nativeCreateInfo.imageFormat = static_cast<VkFormat>(surfaceFormat.format);
    nativeCreateInfo.imageColorSpace = static_cast<VkColorSpaceKHR>(surfaceFormat.colorSpace);
    nativeCreateInfo.imageExtent = extent;
    nativeCreateInfo.imageArrayLayers = 1;
    nativeCreateInfo.imageUsage = static_cast<VkImageUsageFlags>(createInfo.imageUsage);
    nativeCreateInfo.imageSharingMode = separatePresentQueue ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    nativeCreateInfo.queueFamilyIndexCount = separatePresentQueue ? 2U : 0U;
    nativeCreateInfo.pQueueFamilyIndices = separatePresentQueue ? queueFamilyIndices : nullptr;
    nativeCreateInfo.preTransform = static_cast<VkSurfaceTransformFlagBitsKHR>(capabilities.currentTransform);
    nativeCreateInfo.compositeAlpha = chooseCompositeAlpha(capabilities);
    nativeCreateInfo.presentMode = static_cast<VkPresentModeKHR>(presentMode);
    nativeCreateInfo.clipped = VK_TRUE;
    nativeCreateInfo.oldSwapchain = oldSwapchain;

    Resources resources{};
    resources.format = surfaceFormat.format;
    resources.colorSpace = surfaceFormat.colorSpace;
    resources.presentMode = presentMode;
    resources.extent = extent;
    if (vkCreateSwapchainKHR(static_cast<VkDevice>(context.device()), &nativeCreateInfo, nullptr, &resources.swapchain) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan swapchain");
    }

    try {
        uint32_t swapchainImageCount = 0;
        if (vkGetSwapchainImagesKHR(static_cast<VkDevice>(context.device()), resources.swapchain, &swapchainImageCount,
                                    nullptr) != VK_SUCCESS ||
            swapchainImageCount == 0) {
            throw std::runtime_error("Failed to query Vulkan swapchain images");
        }

        std::vector<VkImage> images(swapchainImageCount);
        if (vkGetSwapchainImagesKHR(static_cast<VkDevice>(context.device()), resources.swapchain, &swapchainImageCount,
                                    images.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to retrieve Vulkan swapchain images");
        }

        resources.renderTargets.reserve(images.size());
        for (const VkImage image : images) {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = static_cast<VkFormat>(surfaceFormat.format);
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            RenderTarget target{};
            target.image_ = image;
            target.extent_ = extent;
            target.format_ = surfaceFormat.format;
            if (vkCreateImageView(static_cast<VkDevice>(context.device()), &viewInfo, nullptr, &target.imageView_) !=
                VK_SUCCESS) {
                throw std::runtime_error("Failed to create Vulkan swapchain image view");
            }
            resources.renderTargets.push_back(target);
        }
    } catch (...) {
        for (const RenderTarget& target : resources.renderTargets) {
            if (target.imageView_ != VK_NULL_HANDLE)
                vkDestroyImageView(static_cast<VkDevice>(context.device()), target.imageView_, nullptr);
        }
        vkDestroySwapchainKHR(static_cast<VkDevice>(context.device()), resources.swapchain, nullptr);
        throw;
    }

    return resources;
}

void Swapchain::assign(Resources&& resources) noexcept {
    destroy();
    swapchain_ = std::exchange(resources.swapchain, VK_NULL_HANDLE);
    format_ = resources.format;
    colorSpace_ = resources.colorSpace;
    presentMode_ = resources.presentMode;
    extent_ = resources.extent;
    renderTargets_ = std::move(resources.renderTargets);
}

void Swapchain::destroy() noexcept {
    if (context_ != nullptr) {
        for (const RenderTarget& target : renderTargets_) {
            if (target.imageView_ != VK_NULL_HANDLE)
                vkDestroyImageView(static_cast<VkDevice>(context_->device()), target.imageView_, nullptr);
        }
        if (swapchain_ != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(static_cast<VkDevice>(context_->device()), swapchain_, nullptr);
    }

    swapchain_ = VK_NULL_HANDLE;
    format_ = vk::Format::eUndefined;
    colorSpace_ = vk::ColorSpaceKHR::eSrgbNonlinear;
    presentMode_ = vk::PresentModeKHR::eFifo;
    extent_ = {};
    renderTargets_.clear();
}

bool RenderTarget::isValid() const noexcept {
    return image_ != VK_NULL_HANDLE && imageView_ != VK_NULL_HANDLE;
}

vk::Image RenderTarget::image() const noexcept {
    return vk::Image{image_};
}

VkImage RenderTarget::nativeImage() const noexcept {
    return image_;
}

vk::ImageView RenderTarget::view() const noexcept {
    return vk::ImageView{imageView_};
}

VkImageView RenderTarget::nativeView() const noexcept {
    return imageView_;
}

VkExtent2D RenderTarget::extent() const noexcept {
    return extent_;
}

vk::Format RenderTarget::format() const noexcept {
    return format_;
}

} // namespace vulkan_graphics
