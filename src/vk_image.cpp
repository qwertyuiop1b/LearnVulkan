#include "vk_image.h"

#include <stdexcept>

namespace vk_engine
{
namespace
{
uint32_t FindMemoryType(const VkContext& context, uint32_t filter)
{
    const vk::PhysicalDeviceMemoryProperties properties = context.GetPhysicalDevice().getMemoryProperties();
    for (uint32_t index = 0; index < properties.memoryTypeCount; ++index)
    {
        if ((filter & (1U << index)) != 0 &&
            (properties.memoryTypes[index].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal))
            return index;
    }
    throw std::runtime_error("failed to find device-local image memory");
}
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

VkImage2D::VkImage2D(const VkContext& context, vk::Extent2D extent, std::span<const std::byte> pixels)
{
    VkBuffer staging(context,
                     pixels.size_bytes(),
                     vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    staging.Write(pixels);
    vk::ImageCreateInfo imageInfo{};
    imageInfo.setImageType(vk::ImageType::e2D)
        .setFormat(vk::Format::eR8G8B8A8Srgb)
        .setExtent(vk::Extent3D{extent.width, extent.height, 1})
        .setMipLevels(1)
        .setArrayLayers(1)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setTiling(vk::ImageTiling::eOptimal)
        .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)
        .setInitialLayout(vk::ImageLayout::eUndefined);
    image = vk::raii::Image(context.GetDevice(), imageInfo);
    const vk::MemoryRequirements requirements = image.getMemoryRequirements();
    memory = vk::raii::DeviceMemory(
        context.GetDevice(),
        vk::MemoryAllocateInfo{requirements.size, FindMemoryType(context, requirements.memoryTypeBits)});
    image.bindMemory(*memory, 0);
    vk::raii::CommandPool pool(
        context.GetDevice(),
        vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, context.GetGraphicQueueFamilyIndex()});
    vk::raii::CommandBuffers buffers(context.GetDevice(),
                                     vk::CommandBufferAllocateInfo{*pool, vk::CommandBufferLevel::ePrimary, 1});
    const vk::raii::CommandBuffer& commandBuffer = buffers.front();
    commandBuffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    Barrier(*commandBuffer,
            *image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            {},
            vk::AccessFlagBits::eTransferWrite,
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer);
    commandBuffer.copyBufferToImage(
        staging.GetHandle(),
        *image,
        vk::ImageLayout::eTransferDstOptimal,
        vk::BufferImageCopy{}
            .setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
            .setImageExtent(vk::Extent3D{extent.width, extent.height, 1}));
    Barrier(*commandBuffer,
            *image,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead,
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader);
    commandBuffer.end();
    vk::raii::Fence fence(context.GetDevice(), vk::FenceCreateInfo{});
    context.GetGraphicQueue().submit(vk::SubmitInfo{}.setCommandBuffers(*commandBuffer), *fence);
    (void)context.GetDevice().waitForFences(*fence, VK_TRUE, UINT64_MAX);
    imageView = vk::raii::ImageView(
        context.GetDevice(),
        vk::ImageViewCreateInfo{{},
                                *image,
                                vk::ImageViewType::e2D,
                                vk::Format::eR8G8B8A8Srgb,
                                {},
                                vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
}
} // namespace vk_engine
