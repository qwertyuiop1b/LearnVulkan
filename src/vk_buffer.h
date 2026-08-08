#pragma once

#include "vk_context.h"

#include <cstddef>
#include <span>

#include <vulkan/vulkan_raii.hpp>

namespace vk_engine
{
class Buffer
{
public:
    Buffer(const VkContext& context,
           vk::DeviceSize size,
           vk::BufferUsageFlags usage,
           vk::MemoryPropertyFlags memoryProperties);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;

    void Write(std::span<const std::byte> data);

    vk::Buffer GetHandle() const noexcept
    {
        return buffer;
    }

private:
    const VkContext& context;
    VkBuffer buffer{VK_NULL_HANDLE};
    VmaAllocation allocation{nullptr};
    vk::DeviceSize size{};
};
} // namespace vk_engine
