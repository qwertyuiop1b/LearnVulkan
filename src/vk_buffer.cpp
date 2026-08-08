#include "vk_buffer.h"

#include "vk_utils.h"

#include <cstring>
#include <stdexcept>

namespace vk_engine
{
namespace
{
VmaAllocationCreateInfo BuildAllocationCreateInfo(vk::MemoryPropertyFlags memoryProperties)
{
    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationCreateInfo.requiredFlags = static_cast<VkMemoryPropertyFlags>(memoryProperties);
    if (memoryProperties & vk::MemoryPropertyFlagBits::eHostVisible)
    {
        allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    return allocationCreateInfo;
}
} // namespace

Buffer::Buffer(const VkContext& inContext,
               vk::DeviceSize inSize,
               vk::BufferUsageFlags usage,
               vk::MemoryPropertyFlags memoryProperties)
    : context(inContext), size(inSize)
{
    if (size == 0)
    {
        throw std::invalid_argument("Vulkan buffer size must be greater than zero");
    }

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = static_cast<VkBufferUsageFlags>(usage);
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    const VmaAllocationCreateInfo allocationCreateInfo = BuildAllocationCreateInfo(memoryProperties);
    VK_CHECK(vmaCreateBuffer(context.GetAllocator(), &bufferCreateInfo, &allocationCreateInfo, &buffer, &allocation,
                             nullptr));
}

Buffer::~Buffer()
{
    if (allocation != nullptr)
    {
        vmaDestroyBuffer(context.GetAllocator(), buffer, allocation);
    }
}

void Buffer::Write(std::span<const std::byte> data)
{
    if (data.size_bytes() > size)
    {
        throw std::out_of_range("Vulkan buffer write exceeds the allocated size");
    }
    if (data.empty())
    {
        return;
    }

    void* mapped = nullptr;
    VK_CHECK(vmaMapMemory(context.GetAllocator(), allocation, &mapped));
    std::memcpy(mapped, data.data(), data.size_bytes());
    vmaUnmapMemory(context.GetAllocator(), allocation);
}
} // namespace vk_engine
