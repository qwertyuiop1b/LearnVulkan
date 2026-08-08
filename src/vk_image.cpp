#include "vk_image.h"

#include "vk_utils.h"

#include <stdexcept>
#include <utility>

namespace vk_engine
{
namespace
{
void Barrier(vk::CommandBuffer commandBuffer,
             vk::Image image,
             vk::ImageLayout oldLayout,
             vk::ImageLayout newLayout,
             vk::AccessFlags sourceAccess,
             vk::AccessFlags destinationAccess,
             vk::PipelineStageFlags sourceStage,
             vk::PipelineStageFlags destinationStage)
{
    vk::ImageMemoryBarrier barrier{};
    barrier.setSrcAccessMask(sourceAccess)
        .setDstAccessMask(destinationAccess)
        .setOldLayout(oldLayout)
        .setNewLayout(newLayout)
        .setImage(image)
        .setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
}
} // namespace

Image::Image(const VkContext& inContext, vk::Extent2D extent, std::span<const std::byte> pixels) : context(&inContext)
{
    Buffer staging(inContext,
                   pixels.size_bytes(),
                   vk::BufferUsageFlagBits::eTransferSrc,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    staging.Write(pixels);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imageInfo.extent = VkExtent3D{extent.width, extent.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VK_CHECK(vmaCreateImage(context->GetAllocator(), &imageInfo, &allocationCreateInfo, &image, &allocation, nullptr));

    vk::raii::CommandPool pool(
        context->GetDevice(),
        vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, context->GetGraphicQueueFamilyIndex()});
    vk::raii::CommandBuffers buffers(context->GetDevice(),
                                     vk::CommandBufferAllocateInfo{*pool, vk::CommandBufferLevel::ePrimary, 1});
    const vk::raii::CommandBuffer& commandBuffer = buffers.front();
    commandBuffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    Barrier(*commandBuffer,
            image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            {},
            vk::AccessFlagBits::eTransferWrite,
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer);
    commandBuffer.copyBufferToImage(
        staging.GetHandle(),
        image,
        vk::ImageLayout::eTransferDstOptimal,
        vk::BufferImageCopy{}
            .setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
            .setImageExtent(vk::Extent3D{extent.width, extent.height, 1}));
    Barrier(*commandBuffer,
            image,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead,
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader);
    commandBuffer.end();
    vk::raii::Fence fence(context->GetDevice(), vk::FenceCreateInfo{});
    context->GetGraphicQueue().submit(vk::SubmitInfo{}.setCommandBuffers(*commandBuffer), *fence);
    (void)context->GetDevice().waitForFences(*fence, VK_TRUE, UINT64_MAX);
    imageView = vk::raii::ImageView(
        context->GetDevice(),
        vk::ImageViewCreateInfo{{},
                                image,
                                vk::ImageViewType::e2D,
                                vk::Format::eR8G8B8A8Srgb,
                                {},
                                vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
}

Image::~Image()
{
    Destroy();
}

Image::Image(Image&& other) noexcept
    : context(other.context), image(other.image), allocation(other.allocation), imageView(std::move(other.imageView))
{
    other.context = nullptr;
    other.image = VK_NULL_HANDLE;
    other.allocation = nullptr;
}

Image& Image::operator=(Image&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }
    Destroy();
    context = other.context;
    image = other.image;
    allocation = other.allocation;
    imageView = std::move(other.imageView);
    other.context = nullptr;
    other.image = VK_NULL_HANDLE;
    other.allocation = nullptr;
    return *this;
}

void Image::Destroy()
{
    imageView = nullptr;
    if (context != nullptr && allocation != nullptr)
    {
        vmaDestroyImage(context->GetAllocator(), image, allocation);
    }
    context = nullptr;
    image = VK_NULL_HANDLE;
    allocation = nullptr;
}
} // namespace vk_engine
