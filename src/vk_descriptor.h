#pragma once

#include "vk_context.h"

#include <cstdint>
#include <deque>
#include <span>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

namespace vk_engine
{
/**
 * @brief Builds a descriptor set layout from sequential binding declarations.
 */
class DescriptorLayoutBuilder
{
public:
    DescriptorLayoutBuilder&
    AddBinding(uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags stageFlags, uint32_t count = 1);
    void Clear();
    vk::raii::DescriptorSetLayout Build(const VkContext& context) const;

private:
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
};

/**
 * @brief Allocates descriptor sets from growing pools with common type ratios.
 */
class DescriptorAllocator
{
public:
    struct PoolSizeRatio
    {
        vk::DescriptorType type{};
        float ratio{1.0F};
    };

    explicit DescriptorAllocator(const VkContext& context);
    ~DescriptorAllocator() = default;

    DescriptorAllocator(const DescriptorAllocator&) = delete;
    DescriptorAllocator& operator=(const DescriptorAllocator&) = delete;
    DescriptorAllocator(DescriptorAllocator&&) = delete;
    DescriptorAllocator& operator=(DescriptorAllocator&&) = delete;

    vk::DescriptorSet Allocate(vk::DescriptorSetLayout layout);
    void ResetPools();

private:
    vk::raii::DescriptorPool CreatePool(uint32_t setCount, std::span<const PoolSizeRatio> ratios) const;
    vk::raii::DescriptorPool GetPool();
    vk::DescriptorSet AllocateFromPool(vk::raii::DescriptorPool& pool, vk::DescriptorSetLayout layout);

    const VkContext& context;
    std::vector<PoolSizeRatio> ratios;
    std::vector<vk::raii::DescriptorPool> readyPools;
    std::vector<vk::raii::DescriptorPool> fullPools;
    uint32_t setsPerPool{8};
};

/**
 * @brief Accumulates descriptor writes and applies them to a target set.
 */
class DescriptorWriter
{
public:
    DescriptorWriter& WriteImage(uint32_t binding,
                                 vk::ImageView imageView,
                                 vk::Sampler sampler,
                                 vk::ImageLayout imageLayout,
                                 vk::DescriptorType type);
    DescriptorWriter& WriteBuffer(
        uint32_t binding, vk::Buffer buffer, vk::DeviceSize size, vk::DeviceSize offset, vk::DescriptorType type);
    void Clear();
    void Update(const VkContext& context, vk::DescriptorSet set);

private:
    static vk::WriteDescriptorSet CreateWrite(uint32_t binding, vk::DescriptorType type);

    std::deque<vk::DescriptorImageInfo> imageInfos;
    std::deque<vk::DescriptorBufferInfo> bufferInfos;
    std::vector<vk::WriteDescriptorSet> writes;
};
} // namespace vk_engine
