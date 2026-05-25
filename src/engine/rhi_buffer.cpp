/**
 * @file rhi_buffer.cpp
 * @brief 第62章：GPU 缓冲区封装实现
 *
 * 封装层次：
 *   Buffer        — VkBuffer + VkDeviceMemory RAII（含常驻映射支持）
 *   IndexBuffer   — 索引缓冲区（uint32_t，设备本地）
 *   StagingPool   — 重用 staging buffer 的帧内池
 *
 * 关键点：
 *   - upload() 自动判断是否需要通过 staging buffer 上传
 *   - StagingPool 避免每帧 vkAllocateMemory 带来的 GPU 停顿
 */

#include <vulkan_tutorial/engine/rhi_buffer.hpp>
#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>

#include <cstring>
#include <stdexcept>

namespace engine {

// ─── Buffer：Move 语义 ──────────────────────────────────────────────────────

Buffer::Buffer(Buffer&& o) noexcept
    : dev_(o.dev_), buffer_(o.buffer_), memory_(o.memory_),
      size_(o.size_), mapped_(o.mapped_)
{
    o.dev_    = nullptr;
    o.buffer_ = VK_NULL_HANDLE;
    o.memory_ = VK_NULL_HANDLE;
    o.size_   = 0;
    o.mapped_ = nullptr;
}

Buffer& Buffer::operator=(Buffer&& o) noexcept
{
    if (this != &o) {
        destroy();
        dev_    = o.dev_;
        buffer_ = o.buffer_;
        memory_ = o.memory_;
        size_   = o.size_;
        mapped_ = o.mapped_;
        o.dev_    = nullptr;
        o.buffer_ = VK_NULL_HANDLE;
        o.memory_ = VK_NULL_HANDLE;
        o.size_   = 0;
        o.mapped_ = nullptr;
    }
    return *this;
}

// ─── Buffer::create ─────────────────────────────────────────────────────────

void Buffer::create(RHIDevice& dev, const CreateInfo& ci)
{
    dev_  = &dev;
    size_ = ci.size;

    // 创建 VkBuffer
    VkBufferCreateInfo bufCI{};
    bufCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size        = ci.size;
    bufCI.usage       = ci.usage;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev.device(), &bufCI, nullptr, &buffer_));

    // 分配并绑定设备内存
    VkMemoryRequirements memReqs{};
    vkGetBufferMemoryRequirements(dev.device(), buffer_, &memReqs);

    VkMemoryAllocateInfo allocCI{};
    allocCI.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocCI.allocationSize  = memReqs.size;
    allocCI.memoryTypeIndex = dev.findMemoryType(memReqs.memoryTypeBits, ci.memProps);
    VK_CHECK(vkAllocateMemory(dev.device(), &allocCI, nullptr, &memory_));
    VK_CHECK(vkBindBufferMemory(dev.device(), buffer_, memory_, 0));

    // 若要求常驻映射且内存可被 CPU 访问
    const bool isHostVisible =
        (ci.memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    if (ci.persistentMap && isHostVisible)
        VK_CHECK(vkMapMemory(dev.device(), memory_, 0, ci.size, 0, &mapped_));
}

// ─── Buffer::destroy ────────────────────────────────────────────────────────

void Buffer::destroy()
{
    if (!dev_ || buffer_ == VK_NULL_HANDLE) return;
    if (mapped_) {
        vkUnmapMemory(dev_->device(), memory_);
        mapped_ = nullptr;
    }
    vkDestroyBuffer(dev_->device(), buffer_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    vkFreeMemory(dev_->device(), memory_, nullptr);
    memory_ = VK_NULL_HANDLE;
    size_   = 0;
    dev_    = nullptr;
}

// ─── Buffer::upload ─────────────────────────────────────────────────────────

void Buffer::upload(RHIDevice& dev, const void* data, VkDeviceSize size,
                    VkDeviceSize offset)
{
    if (!buffer_) return;

    // 若已有常驻映射（HOST_VISIBLE），直接 memcpy
    if (mapped_) {
        std::memcpy(static_cast<uint8_t*>(mapped_) + offset, data, size);
        return;
    }

    // 否则通过临时 staging buffer 上传到 DEVICE_LOCAL 内存
    Buffer staging;
    Buffer::CreateInfo stagingCI{};
    stagingCI.size     = size;
    stagingCI.usage    = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingCI.memProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                       | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    staging.create(dev, stagingCI);

    void* stagingMapped = nullptr;
    VK_CHECK(vkMapMemory(dev.device(), staging.memory_, 0, size, 0, &stagingMapped));
    std::memcpy(stagingMapped, data, size);
    vkUnmapMemory(dev.device(), staging.memory_);

    // 一次性命令：从 staging 复制到目标 buffer
    VkCommandBuffer cmd = dev.beginOneShot();
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = offset;
    region.size      = size;
    vkCmdCopyBuffer(cmd, staging.buffer_, buffer_, 1, &region);
    dev.endOneShot(cmd);
}

// ─── Buffer::write ──────────────────────────────────────────────────────────

void Buffer::write(const void* data, VkDeviceSize size, VkDeviceSize offset)
{
    if (!mapped_) return;
    std::memcpy(static_cast<uint8_t*>(mapped_) + offset, data, size);
}

// ─── IndexBuffer::create ────────────────────────────────────────────────────

void IndexBuffer::create(RHIDevice& dev, const std::vector<uint32_t>& indices)
{
    indexCount_ = static_cast<uint32_t>(indices.size());
    const VkDeviceSize bufferSize = sizeof(uint32_t) * indices.size();

    Buffer::CreateInfo ci{};
    ci.size     = bufferSize;
    ci.usage    = VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ci.memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    buf_.create(dev, ci);
    buf_.upload(dev, indices.data(), bufferSize);
}

// ─── StagingPool ─────────────────────────────────────────────────────────────

void StagingPool::create(RHIDevice& dev, VkDeviceSize chunkSize)
{
    dev_       = &dev;
    chunkSize_ = chunkSize;
}

void StagingPool::destroy()
{
    for (auto& buf : pool_)
        buf.destroy();
    pool_.clear();
    inUse_.clear();
    dev_ = nullptr;
}

void StagingPool::reset()
{
    // 帧结束时将所有 staging buffer 标记为可用（不释放内存）
    std::fill(inUse_.begin(), inUse_.end(), false);
}

StagingPool::StagingBuffer StagingPool::acquire(VkDeviceSize size)
{
    // 从池中找一个≥size 且空闲的 buffer
    const VkDeviceSize requiredSize = std::max(size, chunkSize_);
    for (size_t i = 0; i < pool_.size(); ++i) {
        if (!inUse_[i] && pool_[i].size() >= requiredSize) {
            inUse_[i] = true;
            StagingBuffer sb{};
            sb.buf    = &pool_[i];
            sb.mapped = pool_[i].mapped();
            sb.size   = pool_[i].size();
            return sb;
        }
    }

    // 池中没有合适的 buffer，新建一个
    Buffer::CreateInfo ci{};
    ci.size          = requiredSize;
    ci.usage         = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    ci.memProps      = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                     | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    ci.persistentMap = true;
    pool_.emplace_back();
    pool_.back().create(*dev_, ci);
    inUse_.push_back(true);

    StagingBuffer sb{};
    sb.buf    = &pool_.back();
    sb.mapped = pool_.back().mapped();
    sb.size   = pool_.back().size();
    return sb;
}

} // namespace engine
