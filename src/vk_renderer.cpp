#include "vk_renderer.h"

#include <limits>

namespace vk_engine
{
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
      imagesInFlight(swapchain.GetImageCount()), imageLayouts(swapchain.GetImageCount(), vk::ImageLayout::eUndefined)
{
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

void VkRenderer::DrawFrame(const RecordCallback& record)
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

    TransitionToColorAttachment(commandBuffer, swapchain.GetImage(imageIndex), imageLayouts[imageIndex]);
    imageLayouts[imageIndex] = vk::ImageLayout::eColorAttachmentOptimal;

    const RenderTarget renderTarget{swapchain.GetImage(imageIndex),
                                    swapchain.GetImageView(imageIndex),
                                    swapchain.GetExtent(),
                                    swapchain.GetImageFormat()};
    record(commandBuffer, renderTarget);

    TransitionToPresent(commandBuffer, swapchain.GetImage(imageIndex));
    imageLayouts[imageIndex] = vk::ImageLayout::ePresentSrcKHR;

    commandBuffer.end();

    context.GetDevice().resetFences(frameFence);

    const vk::Semaphore waitSemaphore = frame.imageAvailable;
    const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
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

    currentFrame = (currentFrame + 1) % kMaxFramesInFlight;
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
    renderFinished = CreateRenderFinishedSemaphores(context.GetDevice(), swapchain.GetImageCount());
    imagesInFlight.assign(swapchain.GetImageCount(), nullptr);
    imageLayouts.assign(swapchain.GetImageCount(), vk::ImageLayout::eUndefined);
}

void VkRenderer::TransitionToColorAttachment(vk::CommandBuffer commandBuffer,
                                             vk::Image image,
                                             vk::ImageLayout oldLayout) const
{
    vk::ImageMemoryBarrier barrier{};
    barrier.setSrcAccessMask({})
        .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
        .setOldLayout(oldLayout)
        .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(image)
        .setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    const vk::PipelineStageFlags sourceStage = oldLayout == vk::ImageLayout::eUndefined
                                                   ? vk::PipelineStageFlagBits::eTopOfPipe
                                                   : vk::PipelineStageFlagBits::eBottomOfPipe;
    commandBuffer.pipelineBarrier(sourceStage, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, {}, {}, barrier);
}

void VkRenderer::TransitionToPresent(vk::CommandBuffer commandBuffer, vk::Image image) const
{
    vk::ImageMemoryBarrier barrier{};
    barrier.setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
        .setDstAccessMask({})
        .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(image)
        .setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                  vk::PipelineStageFlagBits::eBottomOfPipe,
                                  {},
                                  {},
                                  {},
                                  barrier);
}
} // namespace vk_engine
