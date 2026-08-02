#pragma once

#include "vk_context.h"

#include <cstddef>
#include <span>

#include <vulkan/vulkan_raii.hpp>

namespace vk_engine
{
class VkBuffer
{
public:
    VkBuffer(const VkContext& context,
             vk::DeviceSize size,
             vk::BufferUsageFlags usage,
             vk::MemoryPropertyFlags memoryProperties);
    ~VkBuffer() = default;

    VkBuffer(const VkBuffer&) = delete;
    VkBuffer& operator=(const VkBuffer&) = delete;
    VkBuffer(VkBuffer&&) = delete;
    VkBuffer& operator=(VkBuffer&&) = delete;

    void Write(std::span<const std::byte> data);

    vk::Buffer GetHandle() const noexcept
    {
        return *buffer;
    }

private:
    const VkContext& context;
    vk::raii::DeviceMemory memory{nullptr};
    vk::raii::Buffer buffer{nullptr};
    vk::DeviceSize size{};
};
} // namespace vk_engine
