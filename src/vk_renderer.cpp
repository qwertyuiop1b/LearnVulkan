#include "vk_renderer.h"

#include <limits>

namespace vk_engine
{
namespace
{
constexpr vk::ImageUsageFlags kDrawImageUsage = vk::ImageUsageFlagBits::eColorAttachment |
                                                vk::ImageUsageFlagBits::eTransferSrc |
                                                vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
constexpr vk::Format kDrawImageFormat = vk::Format::eR16G16B16A16Sfloat;

struct AccessStage
{
    vk::AccessFlags access{};
    vk::PipelineStageFlags stage{};
};

AccessStage ResolveDrawImageSourceSync(vk::ImageLayout layout)
{
    if (layout == vk::ImageLayout::eGeneral)
    {
        return {vk::AccessFlagBits::eShaderWrite, vk::PipelineStageFlagBits::eComputeShader};
    }
    return {vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eColorAttachmentOutput};
}

void ExecuteImageBarrier(vk::CommandBuffer commandBuffer,
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
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(image)
        .setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
}
} // namespace

VkFrameContext::VkFrameContext(const vk::raii::Device& device, uint32_t queueFamilyIndex)
{
    vk::CommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
        .setQueueFamilyIndex(queueFamilyIndex);
    commandPool = vk::raii::CommandPool(device, commandPoolCreateInfo);
    vk::CommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.setCommandPool(*commandPool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);
    vk::raii::CommandBuffers commandBuffers(device, commandBufferAllocateInfo);
    commandBuffer = std::move(commandBuffers.front());
    const vk::SemaphoreCreateInfo semaphoreCreateInfo{};
    imageAvailable = vk::raii::Semaphore(device, semaphoreCreateInfo);
    const vk::FenceCreateInfo fenceCreateInfo{vk::FenceCreateFlagBits::eSignaled};
    inFlight = vk::raii::Fence(device, fenceCreateInfo);
}

VkRenderer::VkRenderer(const VkContext& inContext, VkSwapchain& inSwapchain)
    : context(inContext), swapchain(inSwapchain),
      frames{VkFrameContext(context.GetDevice(), context.GetGraphicQueueFamilyIndex()),
             VkFrameContext(context.GetDevice(), context.GetGraphicQueueFamilyIndex())},
      renderFinished(CreateRenderFinishedSemaphores(context.GetDevice(), swapchain.GetImageCount())),
      imagesInFlight(swapchain.GetImageCount()),
      swapchainImageLayouts(swapchain.GetImageCount(), vk::ImageLayout::eUndefined)
{
    CreateDrawImage();
}

VkRenderer::~VkRenderer() noexcept
{
    try
    {
        WaitIdle();
    }
    catch (const vk::SystemError&)
    {
        // Destructors cannot propagate a device-loss error.
    }
}

vk::Format VkRenderer::DrawImageFormat()
{
    return kDrawImageFormat;
}

void VkRenderer::CreateDrawImage()
{
    drawImage = Image(context, swapchain.GetExtent(), kDrawImageFormat, kDrawImageUsage);
    drawImageLayout = vk::ImageLayout::eUndefined;
}

void VkRenderer::DrawFrame(const RenderCallback& callback)
{
    VkFrameContext& frame = frames[currentFrame];
    const vk::Fence frameFence = frame.inFlight;
    (void)context.GetDevice().waitForFences(frameFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    uint32_t imageIndex = 0;
    bool shouldRecreate = false;
    try
    {
        const vk::ResultValue<uint32_t> acquireResult =
            swapchain.GetHandle().acquireNextImage(std::numeric_limits<uint64_t>::max(), frame.imageAvailable, nullptr);
        if (acquireResult.result != vk::Result::eSuccess && acquireResult.result != vk::Result::eSuboptimalKHR)
        {
            return;
        }
        imageIndex = acquireResult.value;
        shouldRecreate = acquireResult.result == vk::Result::eSuboptimalKHR;
    }
    catch (const vk::OutOfDateKHRError&)
    {
        RecreateSwapchain();
        return;
    }
    if (imagesInFlight[imageIndex])
    {
        (void)context.GetDevice().waitForFences(
            imagesInFlight[imageIndex], VK_TRUE, std::numeric_limits<uint64_t>::max());
    }
    imagesInFlight[imageIndex] = frameFence;
    frame.commandPool.reset();
    const vk::CommandBuffer commandBuffer = frame.commandBuffer;
    commandBuffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    RenderHelper helper(commandBuffer,
                        drawImage.GetHandle(),
                        drawImage.GetImageView(),
                        drawImage.GetExtent(),
                        drawImage.GetFormat(),
                        drawImageLayout,
                        currentFrame);
    callback(commandBuffer, helper);
    const AccessStage drawSourceSync = ResolveDrawImageSourceSync(drawImageLayout);
    TransitionImage(commandBuffer,
                    drawImage.GetHandle(),
                    drawImageLayout,
                    vk::ImageLayout::eTransferSrcOptimal,
                    drawSourceSync.access,
                    vk::AccessFlagBits::eTransferRead,
                    drawSourceSync.stage,
                    vk::PipelineStageFlagBits::eTransfer);
    drawImageLayout = vk::ImageLayout::eTransferSrcOptimal;
    const vk::Image swapchainImage = swapchain.GetImage(imageIndex);
    const vk::PipelineStageFlags swapchainSourceStage = swapchainImageLayouts[imageIndex] == vk::ImageLayout::eUndefined
                                                            ? vk::PipelineStageFlagBits::eTopOfPipe
                                                            : vk::PipelineStageFlagBits::eBottomOfPipe;
    TransitionImage(commandBuffer,
                    swapchainImage,
                    swapchainImageLayouts[imageIndex],
                    vk::ImageLayout::eTransferDstOptimal,
                    {},
                    vk::AccessFlagBits::eTransferWrite,
                    swapchainSourceStage,
                    vk::PipelineStageFlagBits::eTransfer);
    swapchainImageLayouts[imageIndex] = vk::ImageLayout::eTransferDstOptimal;
    BlitDrawImageToSwapchain(commandBuffer, swapchainImage);
    TransitionImage(commandBuffer,
                    swapchainImage,
                    swapchainImageLayouts[imageIndex],
                    vk::ImageLayout::ePresentSrcKHR,
                    vk::AccessFlagBits::eTransferWrite,
                    {},
                    vk::PipelineStageFlagBits::eTransfer,
                    vk::PipelineStageFlagBits::eBottomOfPipe);
    swapchainImageLayouts[imageIndex] = vk::ImageLayout::ePresentSrcKHR;
    commandBuffer.end();
    context.GetDevice().resetFences(frameFence);
    const vk::Semaphore waitSemaphore = frame.imageAvailable;
    const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;
    const vk::Semaphore signalSemaphore = renderFinished[imageIndex];
    vk::SubmitInfo submitInfo{};
    submitInfo.setWaitSemaphores(waitSemaphore)
        .setWaitDstStageMask(waitStage)
        .setCommandBuffers(commandBuffer)
        .setSignalSemaphores(signalSemaphore);
    context.GetGraphicQueue().submit(submitInfo, frameFence);
    const vk::SwapchainKHR swapchainHandle = swapchain.GetHandle();
    vk::PresentInfoKHR presentInfo{};
    presentInfo.setWaitSemaphores(signalSemaphore).setSwapchains(swapchainHandle).setImageIndices(imageIndex);
    try
    {
        shouldRecreate =
            shouldRecreate || context.GetPresentQueue().presentKHR(presentInfo) == vk::Result::eSuboptimalKHR;
    }
    catch (const vk::OutOfDateKHRError&)
    {
        shouldRecreate = true;
    }
    if (shouldRecreate)
    {
        RecreateSwapchain();
    }
    currentFrame = (currentFrame + 1) % RenderHelper::kFramesInFlight;
}

void VkRenderer::WaitIdle() const
{
    context.GetDevice().waitIdle();
}

std::vector<vk::raii::Semaphore> VkRenderer::CreateRenderFinishedSemaphores(const vk::raii::Device& device,
                                                                            uint32_t imageCount)
{
    const vk::SemaphoreCreateInfo createInfo{};
    std::vector<vk::raii::Semaphore> semaphores;
    semaphores.reserve(imageCount);
    for (uint32_t index = 0; index < imageCount; ++index)
    {
        semaphores.emplace_back(device, createInfo);
    }
    return semaphores;
}

void VkRenderer::RecreateSwapchain()
{
    if (!swapchain.HasValidFramebufferExtent())
    {
        return;
    }
    WaitIdle();
    swapchain.Recreate();
    CreateDrawImage();
    renderFinished = CreateRenderFinishedSemaphores(context.GetDevice(), swapchain.GetImageCount());
    imagesInFlight.assign(swapchain.GetImageCount(), nullptr);
    swapchainImageLayouts.assign(swapchain.GetImageCount(), vk::ImageLayout::eUndefined);
}

void VkRenderer::TransitionImage(vk::CommandBuffer commandBuffer,
                                 vk::Image image,
                                 vk::ImageLayout oldLayout,
                                 vk::ImageLayout newLayout,
                                 vk::AccessFlags sourceAccess,
                                 vk::AccessFlags destinationAccess,
                                 vk::PipelineStageFlags sourceStage,
                                 vk::PipelineStageFlags destinationStage) const
{
    ExecuteImageBarrier(
        commandBuffer, image, oldLayout, newLayout, sourceAccess, destinationAccess, sourceStage, destinationStage);
}

void VkRenderer::BlitDrawImageToSwapchain(vk::CommandBuffer commandBuffer, vk::Image swapchainImage) const
{
    const vk::Extent2D extent = drawImage.GetExtent();
    const vk::Offset3D extentOffset{static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
    vk::ImageBlit blitRegion{};
    blitRegion.setSrcSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
        .setDstSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    blitRegion.srcOffsets[0] = vk::Offset3D{0, 0, 0};
    blitRegion.srcOffsets[1] = extentOffset;
    blitRegion.dstOffsets[0] = vk::Offset3D{0, 0, 0};
    blitRegion.dstOffsets[1] = extentOffset;
    commandBuffer.blitImage(drawImage.GetHandle(),
                            vk::ImageLayout::eTransferSrcOptimal,
                            swapchainImage,
                            vk::ImageLayout::eTransferDstOptimal,
                            blitRegion,
                            vk::Filter::eNearest);
}

RenderHelper::RenderHelper(vk::CommandBuffer inCommandBuffer,
                           vk::Image inImage,
                           vk::ImageView inImageView,
                           vk::Extent2D inExtent,
                           vk::Format inFormat,
                           vk::ImageLayout& inCurrentLayout,
                           size_t inFrameIndex)
    : commandBuffer(inCommandBuffer), image(inImage), imageView(inImageView), extent(inExtent), format(inFormat),
      currentLayout(inCurrentLayout), frameIndex(inFrameIndex)
{
}

void RenderHelper::TransitionTo(vk::ImageLayout newLayout, vk::AccessFlags dstAccess, vk::PipelineStageFlags dstStage)
{
    if (currentLayout == newLayout)
    {
        return;
    }
    vk::AccessFlags srcAccess{};
    vk::PipelineStageFlags srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
    if (currentLayout != vk::ImageLayout::eUndefined)
    {
        const AccessStage sourceSync = ResolveDrawImageSourceSync(currentLayout);
        srcAccess = sourceSync.access;
        srcStage = sourceSync.stage;
    }
    ExecuteImageBarrier(commandBuffer, image, currentLayout, newLayout, srcAccess, dstAccess, srcStage, dstStage);
    currentLayout = newLayout;
}

void RenderHelper::TransitionToCompute()
{
    TransitionTo(
        vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, vk::PipelineStageFlagBits::eComputeShader);
}

void RenderHelper::TransitionToGraphics()
{
    TransitionTo(vk::ImageLayout::eColorAttachmentOptimal,
                 vk::AccessFlagBits::eColorAttachmentWrite,
                 vk::PipelineStageFlagBits::eColorAttachmentOutput);
}
} // namespace vk_engine
