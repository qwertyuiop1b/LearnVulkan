/**
 * @file command_recorder.cpp
 * @brief 第67章：命令录制封装实现
 *
 * 封装层次说明：
 *   CommandPool      — 每帧独立 pool，支持 RESET_COMMAND_BUFFER_BIT
 *   CommandRecorder  — RAII begin/end，保证 end 一定被调用
 *   RenderPassScope  — RAII beginRenderPass/endRenderPass
 *   BarrierBatch     — 积累多个 barrier，一次 vkCmdPipelineBarrier 提交
 *   DrawCallBatch    — 排序后减少冗余管线/描述符切换
 */

#include <vulkan_tutorial/engine/command_recorder.hpp>
#include <vulkan_tutorial/utils.hpp>

#include <algorithm>
#include <stdexcept>

namespace engine {

// ─── CommandPool ──────────────────────────────────────────────────────────────

void CommandPool::create(RHIDevice& dev, uint32_t frameCount)
{
    dev_ = &dev;
    pools_.resize(frameCount);

    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = dev.graphicsQueue().familyIndex;

    for (auto& pool : pools_)
        VK_CHECK(vkCreateCommandPool(dev.device(), &ci, nullptr, &pool));
}

void CommandPool::destroy()
{
    for (auto pool : pools_)
        if (pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(dev_->device(), pool, nullptr);
    pools_.clear();
}

void CommandPool::reset(uint32_t frameIndex)
{
    VK_CHECK(vkResetCommandPool(dev_->device(), pools_[frameIndex], 0));
}

VkCommandBuffer CommandPool::allocate(uint32_t frameIndex, VkCommandBufferLevel level)
{
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = pools_[frameIndex];
    ai.level              = level;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(dev_->device(), &ai, &cmd));
    return cmd;
}

// ─── CommandRecorder ──────────────────────────────────────────────────────────

CommandRecorder::CommandRecorder(RHIDevice& /*dev*/, VkCommandBuffer cmd, Hint hint)
    : cmd_(cmd), ended_(false)
{
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    switch (hint) {
    case OneShot:
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        break;
    case ReRecordable:
        bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        break;
    default:
        bi.flags = 0;
        break;
    }

    VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
}

CommandRecorder::~CommandRecorder()
{
    if (!ended_ && cmd_ != VK_NULL_HANDLE)
        end();
}

void CommandRecorder::end()
{
    if (ended_) return;
    VK_CHECK(vkEndCommandBuffer(cmd_));
    ended_ = true;
}

void CommandRecorder::bindPipeline(VkPipeline pipeline, VkPipelineBindPoint bp)
{
    vkCmdBindPipeline(cmd_, bp, pipeline);
}

void CommandRecorder::setViewportScissor(float x, float y, float w, float h)
{
    VkViewport vp{x, y, w, h, 0.0f, 1.0f};
    vkCmdSetViewport(cmd_, 0, 1, &vp);

    VkRect2D sc{
        {static_cast<int32_t>(x), static_cast<int32_t>(y)},
        {static_cast<uint32_t>(w), static_cast<uint32_t>(h)}
    };
    vkCmdSetScissor(cmd_, 0, 1, &sc);
}

void CommandRecorder::pushConstants(VkPipelineLayout layout, VkShaderStageFlags stages,
                                    uint32_t offset, uint32_t size, const void* data)
{
    vkCmdPushConstants(cmd_, layout, stages, offset, size, data);
}

// ─── RenderPassScope ──────────────────────────────────────────────────────────

RenderPassScope::RenderPassScope(VkCommandBuffer cmd,
                                 VkRenderPass renderPass,
                                 VkFramebuffer framebuffer,
                                 VkExtent2D extent,
                                 std::initializer_list<VkClearValue> clearValues,
                                 VkSubpassContents contents)
    : cmd_(cmd), ended_(false)
{
    std::vector<VkClearValue> cv(clearValues);

    VkRenderPassBeginInfo bi{};
    bi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    bi.renderPass        = renderPass;
    bi.framebuffer       = framebuffer;
    bi.renderArea.offset = {0, 0};
    bi.renderArea.extent = extent;
    bi.clearValueCount   = static_cast<uint32_t>(cv.size());
    bi.pClearValues      = cv.data();

    vkCmdBeginRenderPass(cmd_, &bi, contents);
}

RenderPassScope::~RenderPassScope()
{
    if (!ended_ && cmd_ != VK_NULL_HANDLE)
        end();
}

RenderPassScope::RenderPassScope(RenderPassScope&& o) noexcept
    : cmd_(o.cmd_), ended_(o.ended_)
{
    o.cmd_   = VK_NULL_HANDLE;
    o.ended_ = true;
}

RenderPassScope& RenderPassScope::operator=(RenderPassScope&& o) noexcept
{
    if (this != &o) {
        if (!ended_ && cmd_ != VK_NULL_HANDLE) end();
        cmd_   = o.cmd_;
        ended_ = o.ended_;
        o.cmd_   = VK_NULL_HANDLE;
        o.ended_ = true;
    }
    return *this;
}

void RenderPassScope::end()
{
    if (ended_) return;
    vkCmdEndRenderPass(cmd_);
    ended_ = true;
}

// ─── BarrierBatch — 访问/阶段 flag 推导 ─────────────────────────────────────

namespace {

struct LayoutAccess {
    VkAccessFlags        access;
    VkPipelineStageFlags stage;
};

LayoutAccess inferAccess(VkImageLayout layout)
{
    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
    case VK_IMAGE_LAYOUT_PREINITIALIZED:
        return { 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT };
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        return { VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL:
        return { VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT };
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        return { VK_ACCESS_SHADER_READ_BIT,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        return { VK_ACCESS_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT };
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        return { VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT };
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        return { VK_ACCESS_MEMORY_READ_BIT,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT };
    default:
        return { VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
    }
}

} // anonymous namespace

// ─── BarrierBatch ─────────────────────────────────────────────────────────────

BarrierBatch& BarrierBatch::imageLayout(VkImage image,
                                         VkImageLayout oldLayout,
                                         VkImageLayout newLayout,
                                         VkImageAspectFlags aspect,
                                         uint32_t mipLevels,
                                         uint32_t layerCount)
{
    auto src = inferAccess(oldLayout);
    auto dst = inferAccess(newLayout);

    srcStage_ |= src.stage;
    dstStage_ |= dst.stage;

    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask       = src.access;
    barrier.dstAccessMask       = dst.access;
    barrier.oldLayout           = oldLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange    = { aspect, 0, mipLevels, 0, layerCount };
    imageBarriers_.push_back(barrier);
    flushed_ = false;
    return *this;
}

BarrierBatch& BarrierBatch::buffer(VkBuffer buf,
                                    VkAccessFlags srcAccess,
                                    VkAccessFlags dstAccess,
                                    VkDeviceSize  offset,
                                    VkDeviceSize  size)
{
    srcStage_ |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    dstStage_ |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    VkBufferMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask       = srcAccess;
    barrier.dstAccessMask       = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer              = buf;
    barrier.offset              = offset;
    barrier.size                = size;
    bufferBarriers_.push_back(barrier);
    flushed_ = false;
    return *this;
}

void BarrierBatch::flush()
{
    if (imageBarriers_.empty() && bufferBarriers_.empty()) {
        flushed_ = true;
        return;
    }

    vkCmdPipelineBarrier(cmd_,
        srcStage_, dstStage_,
        0,
        0, nullptr,
        static_cast<uint32_t>(bufferBarriers_.size()), bufferBarriers_.data(),
        static_cast<uint32_t>(imageBarriers_.size()),  imageBarriers_.data());

    imageBarriers_.clear();
    bufferBarriers_.clear();
    srcStage_ = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dstStage_ = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    flushed_  = true;
}

// ─── DrawCallBatch ────────────────────────────────────────────────────────────

void DrawCallBatch::flush(VkCommandBuffer cmd)
{
    lastPipeline_ = VK_NULL_HANDLE;
    lastDescSet_  = VK_NULL_HANDLE;

    for (const auto& dc : drawCalls_) {
        if (dc.pipeline != lastPipeline_) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, dc.pipeline);
            lastPipeline_ = dc.pipeline;
        }

        if (dc.descSet != VK_NULL_HANDLE && dc.descSet != lastDescSet_) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                dc.layout, 0, 1, &dc.descSet, 0, nullptr);
            lastDescSet_ = dc.descSet;
        }

        if (!dc.pushConstantData.empty()) {
            vkCmdPushConstants(cmd, dc.layout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, static_cast<uint32_t>(dc.pushConstantData.size()),
                dc.pushConstantData.data());
        }

        if (dc.vertexBuf != VK_NULL_HANDLE) {
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &dc.vertexBuf, &offset);
        }

        if (dc.indexBuf != VK_NULL_HANDLE) {
            vkCmdBindIndexBuffer(cmd, dc.indexBuf, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, dc.indexCount, dc.instanceCount, 0, 0, 0);
        } else {
            vkCmdDraw(cmd, dc.vertexCount, dc.instanceCount, 0, 0);
        }
    }
}

} // namespace engine
