#include <graphics/memory/image.hpp>

#include <stdexcept>
#include <utility>

namespace vulkan_graphics {

Image::Image(VulkanAllocator& allocator, const ImageCreateInfo& createInfo) : allocator_(&allocator) {
    if (!allocator.isValid())
        throw std::invalid_argument("Image requires a valid VulkanAllocator");
    if (createInfo.extent.width == 0 || createInfo.extent.height == 0)
        throw std::invalid_argument("Image extent must be greater than zero");
    if (createInfo.format == vk::Format::eUndefined)
        throw std::invalid_argument("Image format must be specified");
    if (createInfo.usage == vk::ImageUsageFlags{})
        throw std::invalid_argument("Image usage flags must not be empty");
    if (createInfo.aspectMask == vk::ImageAspectFlags{})
        throw std::invalid_argument("Image aspect mask must not be empty");
    if (createInfo.mipLevels == 0 || createInfo.arrayLayers == 0)
        throw std::invalid_argument("Image mip levels and array layers must be greater than zero");

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = static_cast<VkFormat>(createInfo.format);
    imageInfo.extent = {createInfo.extent.width, createInfo.extent.height, 1};
    imageInfo.mipLevels = createInfo.mipLevels;
    imageInfo.arrayLayers = createInfo.arrayLayers;
    imageInfo.samples = static_cast<VkSampleCountFlagBits>(createInfo.samples);
    imageInfo.tiling = static_cast<VkImageTiling>(createInfo.tiling);
    imageInfo.usage = static_cast<VkImageUsageFlags>(createInfo.usage);
    imageInfo.sharingMode = static_cast<VkSharingMode>(createInfo.sharingMode);
    imageInfo.queueFamilyIndexCount = static_cast<uint32_t>(createInfo.queueFamilyIndices.size());
    imageInfo.pQueueFamilyIndices = createInfo.queueFamilyIndices.data();
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(allocator.nativeHandle(), &imageInfo, &allocationInfo, &image_, &allocation_, nullptr) !=
        VK_SUCCESS) {
        allocator_ = nullptr;
        throw std::runtime_error("Failed to create Vulkan image");
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image_;
    viewInfo.viewType = createInfo.arrayLayers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = static_cast<VkFormat>(createInfo.format);
    viewInfo.subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(createInfo.aspectMask);
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = createInfo.mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = createInfo.arrayLayers;
    if (vkCreateImageView(static_cast<VkDevice>(allocator.context().device()), &viewInfo, nullptr, &imageView_) !=
        VK_SUCCESS) {
        vmaDestroyImage(allocator.nativeHandle(), image_, allocation_);
        allocator_ = nullptr;
        image_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to create Vulkan image view");
    }

    extent_ = createInfo.extent;
    format_ = createInfo.format;
    usage_ = createInfo.usage;
    aspectMask_ = createInfo.aspectMask;
    mipLevels_ = createInfo.mipLevels;
    arrayLayers_ = createInfo.arrayLayers;
    stateTracker_ = ImageStateTracker{mipLevels_, arrayLayers_};
}

Image::~Image() noexcept {
    destroy();
}

Image::Image(Image&& other) noexcept
    : allocator_(std::exchange(other.allocator_, nullptr)),
      image_(std::exchange(other.image_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      imageView_(std::exchange(other.imageView_, VK_NULL_HANDLE)),
      extent_(std::exchange(other.extent_, {})),
      format_(std::exchange(other.format_, vk::Format::eUndefined)),
      usage_(std::exchange(other.usage_, {})),
      aspectMask_(std::exchange(other.aspectMask_, {})),
      mipLevels_(std::exchange(other.mipLevels_, 0)),
      arrayLayers_(std::exchange(other.arrayLayers_, 0)),
      stateTracker_(std::move(other.stateTracker_)) {
    other.stateTracker_.reset();
}

Image& Image::operator=(Image&& other) noexcept {
    if (this == &other)
        return *this;

    destroy();
    allocator_ = std::exchange(other.allocator_, nullptr);
    image_ = std::exchange(other.image_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    imageView_ = std::exchange(other.imageView_, VK_NULL_HANDLE);
    extent_ = std::exchange(other.extent_, {});
    format_ = std::exchange(other.format_, vk::Format::eUndefined);
    usage_ = std::exchange(other.usage_, {});
    aspectMask_ = std::exchange(other.aspectMask_, {});
    mipLevels_ = std::exchange(other.mipLevels_, 0);
    arrayLayers_ = std::exchange(other.arrayLayers_, 0);
    stateTracker_ = std::move(other.stateTracker_);
    other.stateTracker_.reset();
    return *this;
}

bool Image::isValid() const noexcept {
    return image_ != VK_NULL_HANDLE;
}

vk::Image Image::handle() const noexcept {
    return vk::Image{image_};
}

VkImage Image::nativeHandle() const noexcept {
    return image_;
}

vk::ImageView Image::view() const noexcept {
    return vk::ImageView{imageView_};
}

VkImageView Image::nativeView() const noexcept {
    return imageView_;
}

VkExtent2D Image::extent() const noexcept {
    return extent_;
}

vk::Format Image::format() const noexcept {
    return format_;
}

vk::ImageUsageFlags Image::usage() const noexcept {
    return usage_;
}

vk::ImageAspectFlags Image::aspectMask() const noexcept {
    return aspectMask_;
}

uint32_t Image::mipLevels() const noexcept {
    return mipLevels_;
}

uint32_t Image::arrayLayers() const noexcept {
    return arrayLayers_;
}

vk::ImageLayout Image::layout() const noexcept {
    return stateTracker_.isValid() ? static_cast<vk::ImageLayout>(stateTracker_.state().layout)
                                   : vk::ImageLayout::eUndefined;
}

const ImageSubresourceState& Image::subresourceState(uint32_t mipLevel, uint32_t arrayLayer) const {
    return stateTracker_.state(mipLevel, arrayLayer);
}

void Image::destroy() noexcept {
    if (imageView_ != VK_NULL_HANDLE)
        vkDestroyImageView(static_cast<VkDevice>(allocator_->context().device()), imageView_, nullptr);
    if (image_ != VK_NULL_HANDLE)
        vmaDestroyImage(allocator_->nativeHandle(), image_, allocation_);

    allocator_ = nullptr;
    image_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    imageView_ = VK_NULL_HANDLE;
    extent_ = {};
    format_ = vk::Format::eUndefined;
    usage_ = {};
    aspectMask_ = {};
    mipLevels_ = 0;
    arrayLayers_ = 0;
    stateTracker_.reset();
}

void Image::setState(const ImageSubresourceState& state) {
    stateTracker_.setRange(0, mipLevels_, 0, arrayLayers_, state);
}

} // namespace vulkan_graphics
