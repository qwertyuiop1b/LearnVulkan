/**
 * @file render_graph.cpp
 * @brief RenderGraph 实现
 */

#include <vulkan_tutorial/render_graph.hpp>
#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>

#include <stdexcept>
#include <string>

namespace vulkan_tutorial {

// ─── 公共 API ────────────────────────────────────────────────────────────────

RgTextureHandle RenderGraph::declareTexture(const RgTextureDesc& desc) {
    RgTextureHandle handle{static_cast<uint32_t>(textures_.size())};
    RgTexture tex;
    tex.desc = desc;
    textures_.push_back(tex);
    return handle;
}

void RenderGraph::addGraphicsPass(const char* name,
                                  std::vector<RgTextureHandle> reads,
                                  std::vector<RgTextureHandle> colorWrites,
                                  RgTextureHandle depthWrite,
                                  GraphicsPassFn fn) {
    PassNode node;
    node.type = PassType::Graphics;
    node.name = name;
    node.reads = std::move(reads);
    node.colorWrites = std::move(colorWrites);
    node.depthWrite = depthWrite;
    node.graphicsFn = std::move(fn);
    passes_.push_back(std::move(node));
}

void RenderGraph::addComputePass(const char* name,
                                 std::vector<RgTextureHandle> reads,
                                 std::vector<RgTextureHandle> writes,
                                 ComputePassFn fn) {
    PassNode node;
    node.type = PassType::Compute;
    node.name = name;
    node.reads = std::move(reads);
    node.colorWrites = std::move(writes); // 复用 colorWrites 字段
    node.depthWrite = {};
    node.computeFn = std::move(fn);
    passes_.push_back(std::move(node));
}

void RenderGraph::resetPasses() {
    passes_.clear();
}

// ─── 构建与销毁 ──────────────────────────────────────────────────────────────

void RenderGraph::build(
    VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool cmdPool, VkQueue queue, VkExtent2D extent) {
    buildExtent_ = extent;
    allocateTextures(device, physicalDevice, cmdPool, queue);
    built_ = true;
}

void RenderGraph::resize(
    VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool cmdPool, VkQueue queue, VkExtent2D newExtent) {
    freeTextures(device);
    for (auto& tex : textures_)
        tex.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    buildExtent_ = newExtent;
    allocateTextures(device, physicalDevice, cmdPool, queue);
}

void RenderGraph::destroy(VkDevice device) {
    freeTextures(device);
    built_ = false;
}

// ─── 执行（每帧调用）────────────────────────────────────────────────────────

void RenderGraph::execute(VkCommandBuffer cmd, uint32_t frameIndex) {
    for (auto& pass : passes_) {
        // 确定此 Pass 中每类资源期望的 Layout
        const bool isCompute = (pass.type == PassType::Compute);

        // 读取资源：采样布局（图形）或 GENERAL（计算）
        const VkImageLayout readLayout = isCompute ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // 写入资源：颜色 Attachment（图形）或 GENERAL（计算）
        const VkImageLayout writeLayout =
            isCompute ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        for (auto h : pass.reads)
            transitionImage(cmd, textures_[h.id], readLayout);

        for (auto h : pass.colorWrites)
            transitionImage(cmd, textures_[h.id], writeLayout);

        if (pass.depthWrite.isValid()) {
            transitionImage(cmd, textures_[pass.depthWrite.id], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        }

        // 执行用户回调
        if (isCompute)
            pass.computeFn(cmd, frameIndex);
        else
            pass.graphicsFn(cmd, frameIndex);
    }
}

// ─── 访问器 ──────────────────────────────────────────────────────────────────

VkImageView RenderGraph::getView(RgTextureHandle h) const {
    return textures_.at(h.id).view;
}

VkImage RenderGraph::getImage(RgTextureHandle h) const {
    return textures_.at(h.id).image;
}

VkImageLayout RenderGraph::getLayout(RgTextureHandle h) const {
    return textures_.at(h.id).currentLayout;
}

void RenderGraph::setLayout(RgTextureHandle h, VkImageLayout layout) {
    textures_.at(h.id).currentLayout = layout;
}

// ─── 私有辅助 ────────────────────────────────────────────────────────────────

void RenderGraph::transitionImage(VkCommandBuffer cmd, RgTexture& tex, VkImageLayout target) {
    if (tex.currentLayout == target)
        return;

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = tex.currentLayout;
    barrier.newLayout = target;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = tex.image;
    barrier.subresourceRange.aspectMask = tex.desc.aspect;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    auto setSrcForLayout = [&](VkImageLayout layout) {
        switch (layout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            break;
        case VK_IMAGE_LAYOUT_GENERAL:
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            break;
        default:
            barrier.srcAccessMask = 0;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            break;
        }
    };

    auto setDstForLayout = [&](VkImageLayout layout) {
        switch (layout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            barrier.dstAccessMask =
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            break;
        case VK_IMAGE_LAYOUT_GENERAL:
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            break;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            barrier.dstAccessMask = 0;
            dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            break;
        default:
            barrier.dstAccessMask = 0;
            dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            break;
        }
    };

    setSrcForLayout(tex.currentLayout);
    setDstForLayout(target);

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    tex.currentLayout = target;
}

void RenderGraph::allocateTextures(VkDevice device,
                                   VkPhysicalDevice physicalDevice,
                                   VkCommandPool cmdPool,
                                   VkQueue queue) {
    (void)cmdPool;
    (void)queue; // 暂不需要单次命令

    for (auto& tex : textures_) {
        const uint32_t w = tex.desc.extent.width != 0 ? tex.desc.extent.width : buildExtent_.width;
        const uint32_t h = tex.desc.extent.height != 0 ? tex.desc.extent.height : buildExtent_.height;

        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.extent = {w, h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.format = tex.desc.format;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage = tex.desc.usage;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateImage(device, &ci, nullptr, &tex.image));

        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(device, tex.image, &memReq);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = memReq.size;
        ai.memoryTypeIndex = findMemoryType(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &tex.memory));
        VK_CHECK(vkBindImageMemory(device, tex.image, tex.memory, 0));

        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = tex.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = tex.desc.format;
        vi.subresourceRange.aspectMask = tex.desc.aspect;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(device, &vi, nullptr, &tex.view));

        tex.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

void RenderGraph::freeTextures(VkDevice device) {
    for (auto& tex : textures_) {
        if (tex.view != VK_NULL_HANDLE)
            vkDestroyImageView(device, tex.view, nullptr);
        if (tex.image != VK_NULL_HANDLE)
            vkDestroyImage(device, tex.image, nullptr);
        if (tex.memory != VK_NULL_HANDLE)
            vkFreeMemory(device, tex.memory, nullptr);
        tex.view = VK_NULL_HANDLE;
        tex.image = VK_NULL_HANDLE;
        tex.memory = VK_NULL_HANDLE;
    }
}

} // namespace vulkan_tutorial
