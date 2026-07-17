#include <graphics/upload_context.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace vulkan_graphics {
namespace {

void checkResult(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string("Vulkan upload operation failed: ") + operation);
}

VkDeviceSize resolveCopySize(const Buffer& source,
                             const Buffer& destination,
                             VkDeviceSize bytes,
                             VkDeviceSize sourceOffset,
                             VkDeviceSize destinationOffset) {
    if (sourceOffset > source.size() || destinationOffset > destination.size())
        throw std::out_of_range("Buffer copy offset exceeds the buffer size");

    const VkDeviceSize sourceRemaining = source.size() - sourceOffset;
    const VkDeviceSize destinationRemaining = destination.size() - destinationOffset;
    const VkDeviceSize copySize = bytes == VK_WHOLE_SIZE ? std::min(sourceRemaining, destinationRemaining) : bytes;
    if (copySize == 0 || copySize > sourceRemaining || copySize > destinationRemaining)
        throw std::out_of_range("Buffer copy range exceeds the buffer size");
    return copySize;
}

} // namespace

UploadContext::UploadContext(const VulkanContext& context)
    : context_(&context), commandPool_(context, context.graphicsQueue().familyIndex) {
    if (context.graphicsQueue().handle == vk::Queue{})
        throw std::invalid_argument("UploadContext requires a graphics queue");
    if (!context.synchronization2Enabled())
        throw std::invalid_argument("UploadContext requires Synchronization2 enabled on VulkanContext");
    queue_ = static_cast<VkQueue>(context.graphicsQueue().handle);

    VkFenceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(static_cast<VkDevice>(context.device()), &createInfo, nullptr, &completionFence_) != VK_SUCCESS)
        throw std::runtime_error("Failed to create Vulkan upload fence");
}

UploadContext::~UploadContext() noexcept {
    if (completionFence_ != VK_NULL_HANDLE)
        vkDestroyFence(static_cast<VkDevice>(context_->device()), completionFence_, nullptr);
}

void UploadContext::copyBuffer(const Buffer& source,
                               Buffer& destination,
                               VkDeviceSize bytes,
                               VkDeviceSize sourceOffset,
                               VkDeviceSize destinationOffset) {
    if (!source.isValid() || !destination.isValid())
        throw std::invalid_argument("Buffer copy requires valid source and destination buffers");
    if (!(source.usage() & vk::BufferUsageFlagBits::eTransferSrc))
        throw std::invalid_argument("Source buffer requires transfer-source usage");
    if (!(destination.usage() & vk::BufferUsageFlagBits::eTransferDst))
        throw std::invalid_argument("Destination buffer requires transfer-destination usage");

    const VkDeviceSize copySize = resolveCopySize(source, destination, bytes, sourceOffset, destinationOffset);
    executeAndWait([&](VkCommandBuffer commandBuffer) {
        VkBufferCopy region{};
        region.srcOffset = sourceOffset;
        region.dstOffset = destinationOffset;
        region.size = copySize;
        vkCmdCopyBuffer(commandBuffer, source.nativeHandle(), destination.nativeHandle(), 1, &region);
    });
}

void UploadContext::transitionImageLayout(Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    if (!image.isValid())
        throw std::invalid_argument("Image layout transition requires a valid image");
    const ImageSubresourceState sourceState = image.subresourceState();
    if (sourceState.layout != static_cast<VkImageLayout>(oldLayout) || !image.stateTracker_.hasUniformState(sourceState))
        throw std::logic_error("Image layout transition does not match the tracked layout");
    if (oldLayout == newLayout)
        return;

    ImageSubresourceState destinationState{};
    destinationState.layout = static_cast<VkImageLayout>(newLayout);
    destinationState.queueFamilyIndex = context_->graphicsQueue().familyIndex;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        if (!(image.usage() & vk::ImageUsageFlagBits::eTransferDst))
            throw std::invalid_argument("Transfer-destination layout requires transfer-destination image usage");
        destinationState.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        destinationState.accessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
               newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        if (!(image.usage() & vk::ImageUsageFlagBits::eSampled))
            throw std::invalid_argument("Shader-read layout requires sampled image usage");
        destinationState.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        destinationState.accessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    } else {
        throw std::invalid_argument("Unsupported image layout transition");
    }

    executeAndWait([&](VkCommandBuffer commandBuffer) {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = sourceState.stageMask;
        barrier.srcAccessMask = sourceState.accessMask;
        barrier.dstStageMask = destinationState.stageMask;
        barrier.dstAccessMask = destinationState.accessMask;
        barrier.oldLayout = static_cast<VkImageLayout>(oldLayout);
        barrier.newLayout = static_cast<VkImageLayout>(newLayout);
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.nativeHandle();
        barrier.subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(image.aspectMask());
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = image.mipLevels();
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = image.arrayLayers();

        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    });
    image.setState(destinationState);
}

void UploadContext::copyBufferToImage(const Buffer& source, Image& destination) {
    if (!source.isValid() || !destination.isValid())
        throw std::invalid_argument("Buffer-to-image copy requires valid source and destination resources");
    if (!(source.usage() & vk::BufferUsageFlagBits::eTransferSrc))
        throw std::invalid_argument("Buffer-to-image copy requires transfer-source buffer usage");
    if (!(destination.usage() & vk::ImageUsageFlagBits::eTransferDst))
        throw std::invalid_argument("Buffer-to-image copy requires transfer-destination image usage");
    if (destination.layout() != vk::ImageLayout::eTransferDstOptimal)
        throw std::logic_error("Buffer-to-image copy requires transfer-destination image layout");
    if (destination.mipLevels() != 1)
        throw std::invalid_argument("Buffer-to-image copy currently supports single-mip images only");

    executeAndWait([&](VkCommandBuffer commandBuffer) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = static_cast<VkImageAspectFlags>(destination.aspectMask());
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = destination.arrayLayers();
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {destination.extent().width, destination.extent().height, 1};
        vkCmdCopyBufferToImage(commandBuffer,
                               source.nativeHandle(),
                               destination.nativeHandle(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1,
                               &region);
    });
}

void UploadContext::executeAndWait(const std::function<void(VkCommandBuffer)>& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    const VkCommandBuffer commandBuffer = allocateCommandBuffer();
    bool submitted = false;
    try {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkResult(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
        record(commandBuffer);
        checkResult(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
        submitAndWait(commandBuffer, submitted);
    } catch (...) {
        if (submitted) {
            try {
                context_->waitIdle();
            } catch (...) {
            }
        }
        freeCommandBuffer(commandBuffer);
        throw;
    }

    freeCommandBuffer(commandBuffer);
}

VkCommandBuffer UploadContext::allocateCommandBuffer() {
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandPool_.nativeHandle();
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    checkResult(vkAllocateCommandBuffers(static_cast<VkDevice>(context_->device()), &allocateInfo, &commandBuffer),
                "vkAllocateCommandBuffers");
    return commandBuffer;
}

void UploadContext::submitAndWait(VkCommandBuffer commandBuffer, bool& submitted) {
    checkResult(vkResetFences(static_cast<VkDevice>(context_->device()), 1, &completionFence_), "vkResetFences");

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = commandBuffer;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    checkResult(vkQueueSubmit2(queue_, 1, &submitInfo, completionFence_), "vkQueueSubmit2");
    submitted = true;
    checkResult(vkWaitForFences(static_cast<VkDevice>(context_->device()), 1, &completionFence_, VK_TRUE,
                                std::numeric_limits<uint64_t>::max()),
                "vkWaitForFences");
}

void UploadContext::freeCommandBuffer(VkCommandBuffer commandBuffer) noexcept {
    if (commandBuffer != VK_NULL_HANDLE)
        vkFreeCommandBuffers(static_cast<VkDevice>(context_->device()), commandPool_.nativeHandle(), 1, &commandBuffer);
}

} // namespace vulkan_graphics
