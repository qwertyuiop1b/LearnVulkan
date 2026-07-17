#pragma once

#include <graphics/memory/vulkan_allocator.hpp>

#include <cstdint>
#include <vector>

namespace vulkan_graphics {

enum class BufferMemoryUsage {
    GpuOnly,
    CpuToGpu,
    GpuToCpu,
};

enum class BufferHostAccess {
    None,
    SequentialWrite,
    Random,
};

struct BufferCreateInfo {
    VkDeviceSize size = 0;
    vk::BufferUsageFlags usage{};
    BufferMemoryUsage memoryUsage = BufferMemoryUsage::GpuOnly;
    BufferHostAccess hostAccess = BufferHostAccess::None;
    bool persistentMap = false;
    vk::SharingMode sharingMode = vk::SharingMode::eExclusive;
    std::vector<uint32_t> queueFamilyIndices;
};

class Buffer final {
  public:
    Buffer() = default;
    Buffer(VulkanAllocator& allocator, const BufferCreateInfo& createInfo);
    ~Buffer() noexcept;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] bool isPersistentlyMapped() const noexcept;
    [[nodiscard]] vk::Buffer handle() const noexcept;
    [[nodiscard]] VkBuffer nativeHandle() const noexcept;
    [[nodiscard]] VkDeviceSize size() const noexcept;
    [[nodiscard]] vk::BufferUsageFlags usage() const noexcept;
    [[nodiscard]] void* mappedData() const noexcept;

    void write(const void* source, VkDeviceSize bytes, VkDeviceSize offset = 0);
    void flush(VkDeviceSize bytes = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
    void invalidate(VkDeviceSize bytes = VK_WHOLE_SIZE, VkDeviceSize offset = 0);

  private:
    void destroy() noexcept;
    void validateRange(VkDeviceSize bytes, VkDeviceSize offset) const;
    void requireHostAccess(const char* operation) const;

    VulkanAllocator* allocator_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    vk::BufferUsageFlags usage_{};
    bool hostAccessible_ = false;
    void* persistentMappedData_ = nullptr;
};

} // namespace vulkan_graphics
