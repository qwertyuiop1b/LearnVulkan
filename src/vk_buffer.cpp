#include "vk_buffer.h"

#include <cstring>
#include <stdexcept>

namespace vk_engine
{
namespace
{
uint32_t FindMemoryType(const VkContext& context, uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
    const vk::PhysicalDeviceMemoryProperties memoryProperties = context.GetPhysicalDevice().getMemoryProperties();
    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index)
    {
        const bool typeSupported = (typeFilter & (1U << index)) != 0;
        const bool propertiesSupported = (memoryProperties.memoryTypes[index].propertyFlags & properties) == properties;
        if (typeSupported && propertiesSupported)
        {
            return index;
        }
    }

    throw std::runtime_error("failed to find a compatible Vulkan memory type");
}
} // namespace

VkBuffer::VkBuffer(const VkContext& inContext,
                   vk::DeviceSize inSize,
                   vk::BufferUsageFlags usage,
                   vk::MemoryPropertyFlags memoryProperties)
    : context(inContext), size(inSize)
{
    if (size == 0)
    {
        throw std::invalid_argument("Vulkan buffer size must be greater than zero");
    }

    vk::BufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.setSize(size).setUsage(usage).setSharingMode(vk::SharingMode::eExclusive);
    buffer = vk::raii::Buffer(context.GetDevice(), bufferCreateInfo);

    const vk::MemoryRequirements requirements = buffer.getMemoryRequirements();
    const uint32_t memoryTypeIndex = FindMemoryType(context, requirements.memoryTypeBits, memoryProperties);

    vk::MemoryAllocateInfo allocateInfo{};
    allocateInfo.setAllocationSize(requirements.size).setMemoryTypeIndex(memoryTypeIndex);
    memory = vk::raii::DeviceMemory(context.GetDevice(), allocateInfo);
    buffer.bindMemory(*memory, 0);
}

void VkBuffer::Write(std::span<const std::byte> data)
{
    if (data.size_bytes() > size)
    {
        throw std::out_of_range("Vulkan buffer write exceeds the allocated size");
    }
    if (data.empty())
    {
        return;
    }

    void* mapped = memory.mapMemory(0, data.size_bytes());
    std::memcpy(mapped, data.data(), data.size_bytes());
    memory.unmapMemory();
}
} // namespace vk_engine
