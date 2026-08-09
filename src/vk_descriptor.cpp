#include "vk_descriptor.h"

#include <algorithm>

namespace vk_engine
{
namespace
{
constexpr uint32_t kMaxSetsPerPool = 4096;
} // namespace

DescriptorLayoutBuilder& DescriptorLayoutBuilder::AddBinding(uint32_t binding,
                                                             vk::DescriptorType type,
                                                             vk::ShaderStageFlags stageFlags,
                                                             uint32_t count)
{
    vk::DescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.setBinding(binding).setDescriptorType(type).setDescriptorCount(count).setStageFlags(stageFlags);
    bindings.push_back(layoutBinding);
    return *this;
}

void DescriptorLayoutBuilder::Clear()
{
    bindings.clear();
}

vk::raii::DescriptorSetLayout DescriptorLayoutBuilder::Build(const VkContext& context) const
{
    vk::DescriptorSetLayoutCreateInfo createInfo{};
    createInfo.setBindings(bindings);
    return vk::raii::DescriptorSetLayout(context.GetDevice(), createInfo);
}

DescriptorAllocator::DescriptorAllocator(const VkContext& inContext) : context(inContext)
{
    ratios = {
        PoolSizeRatio{vk::DescriptorType::eUniformBuffer, 4.0F},
        PoolSizeRatio{vk::DescriptorType::eStorageBuffer, 2.0F},
        PoolSizeRatio{vk::DescriptorType::eSampler, 2.0F},
        PoolSizeRatio{vk::DescriptorType::eSampledImage, 4.0F},
        PoolSizeRatio{vk::DescriptorType::eCombinedImageSampler, 4.0F},
        PoolSizeRatio{vk::DescriptorType::eStorageImage, 2.0F},
    };
}

vk::raii::DescriptorPool DescriptorAllocator::CreatePool(uint32_t setCount,
                                                         std::span<const PoolSizeRatio> poolRatios) const
{
    std::vector<vk::DescriptorPoolSize> poolSizes;
    poolSizes.reserve(poolRatios.size());
    for (const PoolSizeRatio& ratio : poolRatios)
    {
        const uint32_t descriptorCount = static_cast<uint32_t>(ratio.ratio * static_cast<float>(setCount));
        poolSizes.push_back(vk::DescriptorPoolSize{ratio.type, std::max(descriptorCount, 1U)});
    }
    vk::DescriptorPoolCreateInfo createInfo{};
    createInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
        .setMaxSets(setCount)
        .setPoolSizes(poolSizes);
    return vk::raii::DescriptorPool(context.GetDevice(), createInfo);
}

vk::raii::DescriptorPool DescriptorAllocator::GetPool()
{
    if (!readyPools.empty())
    {
        vk::raii::DescriptorPool pool = std::move(readyPools.back());
        readyPools.pop_back();
        return pool;
    }
    vk::raii::DescriptorPool pool = CreatePool(setsPerPool, ratios);
    setsPerPool = std::min(setsPerPool * 2, kMaxSetsPerPool);
    return pool;
}

vk::DescriptorSet DescriptorAllocator::AllocateFromPool(vk::raii::DescriptorPool& pool, vk::DescriptorSetLayout layout)
{
    vk::DescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.setDescriptorPool(*pool).setSetLayouts(layout);
    const vk::Device device = *context.GetDevice();
    const std::vector<vk::DescriptorSet> allocated = device.allocateDescriptorSets(allocateInfo);
    readyPools.push_back(std::move(pool));
    return allocated.front();
}

vk::DescriptorSet DescriptorAllocator::Allocate(vk::DescriptorSetLayout layout)
{
    vk::raii::DescriptorPool pool = GetPool();
    try
    {
        return AllocateFromPool(pool, layout);
    }
    catch (const vk::SystemError&)
    {
        fullPools.push_back(std::move(pool));
        pool = GetPool();
        return AllocateFromPool(pool, layout);
    }
}

void DescriptorAllocator::ResetPools()
{
    const vk::Device device = *context.GetDevice();
    for (vk::raii::DescriptorPool& pool : readyPools)
    {
        device.resetDescriptorPool(*pool);
    }
    for (vk::raii::DescriptorPool& pool : fullPools)
    {
        device.resetDescriptorPool(*pool);
        readyPools.push_back(std::move(pool));
    }
    fullPools.clear();
}

vk::WriteDescriptorSet DescriptorWriter::CreateWrite(uint32_t binding, vk::DescriptorType type)
{
    vk::WriteDescriptorSet write{};
    write.setDstBinding(binding).setDescriptorCount(1).setDescriptorType(type);
    return write;
}

DescriptorWriter& DescriptorWriter::WriteImage(uint32_t binding,
                                               vk::ImageView imageView,
                                               vk::Sampler sampler,
                                               vk::ImageLayout imageLayout,
                                               vk::DescriptorType type)
{
    imageInfos.push_back(vk::DescriptorImageInfo{sampler, imageView, imageLayout});
    vk::WriteDescriptorSet write = CreateWrite(binding, type);
    write.setImageInfo(imageInfos.back());
    writes.push_back(write);
    return *this;
}

DescriptorWriter& DescriptorWriter::WriteBuffer(
    uint32_t binding, vk::Buffer buffer, vk::DeviceSize size, vk::DeviceSize offset, vk::DescriptorType type)
{
    bufferInfos.push_back(vk::DescriptorBufferInfo{buffer, offset, size});
    vk::WriteDescriptorSet write = CreateWrite(binding, type);
    write.setBufferInfo(bufferInfos.back());
    writes.push_back(write);
    return *this;
}

void DescriptorWriter::Clear()
{
    imageInfos.clear();
    bufferInfos.clear();
    writes.clear();
}

void DescriptorWriter::Update(const VkContext& context, vk::DescriptorSet set)
{
    for (vk::WriteDescriptorSet& write : writes)
    {
        write.setDstSet(set);
    }
    context.GetDevice().updateDescriptorSets(writes, {});
}
} // namespace vk_engine
