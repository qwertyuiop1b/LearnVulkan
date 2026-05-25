#pragma once
/**
 * @file rhi_buffer.hpp
 * @brief 第62章：GPU 缓冲区封装
 *
 * 封装层次：
 *   Buffer         — 最基础的 VkBuffer + VkDeviceMemory RAII 包装
 *   VertexBuffer   — 类型化顶点缓冲区
 *   IndexBuffer    — 索引缓冲区
 *   UniformBuffer  — 模板化的统一缓冲区（双缓冲）
 *   StagingPool    — 重用 staging buffer 的池（减少 vkAllocateMemory 次数）
 */

#include "rhi_device.hpp"
#include <cstring>
#include <vector>

namespace engine {

// ─── 基础缓冲区 ─────────────────────────────────────────────────────────────

/**
 * @brief GPU 缓冲区（RAII）
 *
 * 设计决策：
 *   - 不继承，用组合描述更复杂的缓冲区类型
 *   - mapped() 返回 CPU 可写指针，仅对 HOST_VISIBLE 分配有效
 *   - 支持 move，禁止 copy（Rule of Five）
 */
class Buffer {
public:
    Buffer() = default;
    ~Buffer() { destroy(); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;

    struct CreateInfo {
        VkDeviceSize          size        = 0;
        VkBufferUsageFlags    usage       = 0;
        VkMemoryPropertyFlags memProps    = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        bool                  persistentMap = false;  ///< HOST_VISIBLE 时保持常驻映射
    };

    void create(RHIDevice& dev, const CreateInfo& ci);
    void destroy();

    /// 上传数据（内部自动使用 staging buffer，适合 DEVICE_LOCAL 目标）
    void upload(RHIDevice& dev, const void* data, VkDeviceSize size,
                VkDeviceSize offset = 0);

    /// 直接写（仅对 HOST_VISIBLE / 常驻映射有效）
    void write(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);

    [[nodiscard]] VkBuffer     handle()  const { return buffer_; }
    [[nodiscard]] VkDeviceSize size()    const { return size_; }
    [[nodiscard]] void*        mapped()  const { return mapped_; }
    [[nodiscard]] bool         isValid() const { return buffer_ != VK_NULL_HANDLE; }

    operator VkBuffer() const { return buffer_; }

private:
    RHIDevice*     dev_    = nullptr;
    VkBuffer       buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize   size_   = 0;
    void*          mapped_ = nullptr;
};

// ─── 类型化缓冲区 ─────────────────────────────────────────────────────────

/// 顶点缓冲区（设备本地，从 CPU 一次性上传）
class VertexBuffer {
public:
    template<typename T>
    void create(RHIDevice& dev, const std::vector<T>& vertices)
    {
        Buffer::CreateInfo ci{};
        ci.size  = sizeof(T) * vertices.size();
        ci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buf_.create(dev, ci);
        buf_.upload(dev, vertices.data(), ci.size);
        vertCount_ = static_cast<uint32_t>(vertices.size());
    }
    void destroy() { buf_.destroy(); }

    void bind(VkCommandBuffer cmd, uint32_t binding = 0, VkDeviceSize offset = 0) const
    {
        VkBuffer b = buf_.handle();
        vkCmdBindVertexBuffers(cmd, binding, 1, &b, &offset);
    }
    void draw(VkCommandBuffer cmd, uint32_t instanceCount = 1, uint32_t firstVertex = 0) const
    {
        vkCmdDraw(cmd, vertCount_, instanceCount, firstVertex, 0);
    }
    [[nodiscard]] uint32_t vertexCount() const { return vertCount_; }
    [[nodiscard]] bool     isValid()     const { return buf_.isValid(); }

private:
    Buffer   buf_;
    uint32_t vertCount_ = 0;
};

/// 索引缓冲区
class IndexBuffer {
public:
    void create(RHIDevice& dev, const std::vector<uint32_t>& indices);
    void destroy() { buf_.destroy(); }

    void bind(VkCommandBuffer cmd) const
    {
        vkCmdBindIndexBuffer(cmd, buf_.handle(), 0, VK_INDEX_TYPE_UINT32);
    }
    void drawIndexed(VkCommandBuffer cmd, uint32_t instanceCount = 1) const
    {
        vkCmdDrawIndexed(cmd, indexCount_, instanceCount, 0, 0, 0);
    }
    [[nodiscard]] uint32_t indexCount() const { return indexCount_; }

private:
    Buffer   buf_;
    uint32_t indexCount_ = 0;
};

// ─── 统一缓冲区（双缓冲）────────────────────────────────────────────────────

/**
 * @brief 类型安全的 UBO，支持每帧更新（双缓冲）
 *
 * 用法：
 * @code
 *   UniformBuffer<CameraData> cameraUBO;
 *   cameraUBO.create(dev, MAX_FRAMES);
 *   // 每帧：
 *   cameraUBO.update(frameIndex, { view, proj, camPos });
 * @endcode
 */
template<typename T>
class UniformBuffer {
public:
    void create(RHIDevice& dev, uint32_t frameCount)
    {
        dev_ = &dev;
        bufs_.resize(frameCount);
        for (auto& b : bufs_) {
            Buffer::CreateInfo ci{};
            ci.size         = sizeof(T);
            ci.usage        = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            ci.memProps     = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            ci.persistentMap = true;
            b.create(dev, ci);
        }
    }

    void destroy() { for (auto& b : bufs_) b.destroy(); bufs_.clear(); }

    void update(uint32_t frameIndex, const T& data)
    {
        bufs_[frameIndex].write(&data, sizeof(T));
    }

    [[nodiscard]] VkBuffer     handle(uint32_t fi)     const { return bufs_[fi].handle(); }
    [[nodiscard]] VkDeviceSize size()                  const { return sizeof(T); }
    [[nodiscard]] uint32_t     frameCount()            const { return static_cast<uint32_t>(bufs_.size()); }

    VkDescriptorBufferInfo descriptorInfo(uint32_t fi) const
    {
        return {bufs_[fi].handle(), 0, sizeof(T)};
    }

private:
    RHIDevice*          dev_ = nullptr;
    std::vector<Buffer> bufs_;
};

// ─── StagingPool ────────────────────────────────────────────────────────────

/**
 * @brief Staging Buffer 池 —— 避免每次上传都分配新内存
 *
 * 工作方式：
 *   1. acquire(size)  — 从池中取出一个≥size 的 staging buffer
 *   2. 写入 CPU 数据  — 通过 mapped()
 *   3. 提交 GPU 复制命令
 *   4. reset()        — 帧结束后归还全部 buffer（不 free，下帧复用）
 */
class StagingPool {
public:
    void create(RHIDevice& dev, VkDeviceSize chunkSize = 64 * 1024 * 1024);
    void destroy();
    void reset();   ///< 帧结束时调用，归还所有 staging buffer

    struct StagingBuffer {
        Buffer*      buf    = nullptr;
        void*        mapped = nullptr;
        VkDeviceSize size   = 0;
    };
    [[nodiscard]] StagingBuffer acquire(VkDeviceSize size);

private:
    RHIDevice*          dev_       = nullptr;
    VkDeviceSize        chunkSize_ = 0;
    std::vector<Buffer> pool_;
    std::vector<bool>   inUse_;
};

} // namespace engine
