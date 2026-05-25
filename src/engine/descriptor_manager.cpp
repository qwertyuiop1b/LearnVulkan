/**
 * @file descriptor_manager.cpp
 * @brief 第66章：DescriptorLayoutCache + DescriptorAllocator + DescriptorBuilder 实现
 *
 * 核心思想：
 *   传统写法每个 descriptor set 需要 ~30 行 Vulkan API 调用。
 *   这三个类将其压缩为 4 行流式调用，同时自动管理 pool 扩容和 layout 去重。
 */

#include <vulkan_tutorial/engine/descriptor_manager.hpp>
#include <vulkan_tutorial/utils.hpp>

#include <algorithm>
#include <stdexcept>

namespace engine {

// ─── DescriptorLayoutCache::LayoutKey ────────────────────────────────────────

bool DescriptorLayoutCache::LayoutKey::operator==(const LayoutKey& o) const
{
    if (bindings.size() != o.bindings.size()) return false;
    for (size_t i = 0; i < bindings.size(); ++i) {
        const auto& a = bindings[i];
        const auto& b = o.bindings[i];
        if (a.binding         != b.binding         ||
            a.descriptorType  != b.descriptorType  ||
            a.stageFlags      != b.stageFlags      ||
            a.descriptorCount != b.descriptorCount)
            return false;
    }
    return true;
}

size_t DescriptorLayoutCache::LayoutKeyHash::operator()(const LayoutKey& k) const
{
    size_t seed = k.bindings.size();
    for (const auto& b : k.bindings) {
        auto mix = [&](size_t v) {
            seed ^= v + 0x9e3779b9ull + (seed << 6) + (seed >> 2);
        };
        mix(std::hash<uint32_t>{}(b.binding));
        mix(std::hash<uint32_t>{}(static_cast<uint32_t>(b.descriptorType)));
        mix(std::hash<uint32_t>{}(b.stageFlags));
        mix(std::hash<uint32_t>{}(b.descriptorCount));
    }
    return seed;
}

// ─── DescriptorLayoutCache ────────────────────────────────────────────────────

void DescriptorLayoutCache::destroy()
{
    for (auto& [key, layout] : cache_)
        vkDestroyDescriptorSetLayout(dev_->device(), layout, nullptr);
    cache_.clear();
}

VkDescriptorSetLayout DescriptorLayoutCache::getOrCreate(
    const std::vector<VkDescriptorSetLayoutBinding>& bindings)
{
    LayoutKey key{bindings};
    auto it = cache_.find(key);
    if (it != cache_.end())
        return it->second;

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings    = bindings.data();

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(dev_->device(), &ci, nullptr, &layout));
    cache_[key] = layout;
    return layout;
}

// ─── DescriptorAllocator ──────────────────────────────────────────────────────

void DescriptorAllocator::init(RHIDevice& dev)
{
    dev_ = &dev;
    currentPool_ = createPool();
    allPools_.push_back(currentPool_);
}

void DescriptorAllocator::destroy()
{
    for (auto pool : allPools_)
        vkDestroyDescriptorPool(dev_->device(), pool, nullptr);
    allPools_.clear();
    usedPools_.clear();
    freePools_.clear();
    currentPool_    = VK_NULL_HANDLE;
    totalAllocated_ = 0;
}

void DescriptorAllocator::reset()
{
    for (auto pool : usedPools_) {
        vkResetDescriptorPool(dev_->device(), pool, 0);
        freePools_.push_back(pool);
    }
    usedPools_.clear();

    if (currentPool_ != VK_NULL_HANDLE) {
        vkResetDescriptorPool(dev_->device(), currentPool_, 0);
        freePools_.push_back(currentPool_);
        currentPool_ = VK_NULL_HANDLE;
    }
    totalAllocated_ = 0;
    currentPool_    = grabPool();
}

VkDescriptorPool DescriptorAllocator::createPool()
{
    constexpr uint32_t COUNT = 1000;
    const VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER,                COUNT },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, COUNT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          COUNT },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          COUNT },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   COUNT },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,   COUNT },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         COUNT },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         COUNT },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, COUNT },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, COUNT },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,        COUNT },
    };

    VkDescriptorPoolCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets       = COUNT * static_cast<uint32_t>(std::size(sizes));
    ci.poolSizeCount = static_cast<uint32_t>(std::size(sizes));
    ci.pPoolSizes    = sizes;

    VkDescriptorPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(dev_->device(), &ci, nullptr, &pool));
    return pool;
}

VkDescriptorPool DescriptorAllocator::grabPool()
{
    if (!freePools_.empty()) {
        VkDescriptorPool pool = freePools_.back();
        freePools_.pop_back();
        return pool;
    }
    VkDescriptorPool pool = createPool();
    allPools_.push_back(pool);
    return pool;
}

VkDescriptorSet DescriptorAllocator::allocate(VkDescriptorSetLayout layout)
{
    if (currentPool_ == VK_NULL_HANDLE)
        currentPool_ = grabPool();

    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = currentPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &layout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult result = vkAllocateDescriptorSets(dev_->device(), &ai, &set);

    if (result == VK_SUCCESS) {
        ++totalAllocated_;
        return set;
    }

    if (result == VK_ERROR_FRAGMENTED_POOL || result == VK_ERROR_OUT_OF_POOL_MEMORY) {
        usedPools_.push_back(currentPool_);
        currentPool_      = grabPool();
        ai.descriptorPool = currentPool_;
        VK_CHECK(vkAllocateDescriptorSets(dev_->device(), &ai, &set));
        ++totalAllocated_;
        return set;
    }

    VK_CHECK(result);
    return VK_NULL_HANDLE;
}

// ─── DescriptorBuilder ────────────────────────────────────────────────────────

DescriptorBuilder& DescriptorBuilder::bindBuffer(
    uint32_t binding,
    const VkDescriptorBufferInfo& info,
    VkDescriptorType type,
    VkShaderStageFlags stages)
{
    bufInfos_.push_back(info);

    VkDescriptorSetLayoutBinding b{};
    b.binding         = binding;
    b.descriptorType  = type;
    b.descriptorCount = 1;
    b.stageFlags      = stages;
    bindings_.push_back(b);

    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstBinding      = binding;
    w.descriptorCount = 1;
    w.descriptorType  = type;
    w.pBufferInfo     = &bufInfos_.back();
    writes_.push_back(w);
    return *this;
}

DescriptorBuilder& DescriptorBuilder::bindImage(
    uint32_t binding,
    const VkDescriptorImageInfo& info,
    VkDescriptorType type,
    VkShaderStageFlags stages)
{
    imgInfos_.push_back(info);

    VkDescriptorSetLayoutBinding b{};
    b.binding         = binding;
    b.descriptorType  = type;
    b.descriptorCount = 1;
    b.stageFlags      = stages;
    bindings_.push_back(b);

    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstBinding      = binding;
    w.descriptorCount = 1;
    w.descriptorType  = type;
    w.pImageInfo      = &imgInfos_.back();
    writes_.push_back(w);
    return *this;
}

DescriptorBuilder& DescriptorBuilder::bindStorageBuffer(
    uint32_t binding,
    const VkDescriptorBufferInfo& info,
    VkShaderStageFlags stages)
{
    return bindBuffer(binding, info, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stages);
}

DescriptorBuilder& DescriptorBuilder::bindStorageImage(
    uint32_t binding,
    const VkDescriptorImageInfo& info,
    VkShaderStageFlags stages)
{
    return bindImage(binding, info, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stages);
}

bool DescriptorBuilder::build(VkDescriptorSet& outSet, VkDescriptorSetLayout& outLayout)
{
    outLayout = cache_->getOrCreate(bindings_);
    outSet    = alloc_->allocate(outLayout);
    if (outSet == VK_NULL_HANDLE) return false;
    return write(outSet);
}

bool DescriptorBuilder::write(VkDescriptorSet set)
{
    for (auto& w : writes_)
        w.dstSet = set;

    vkUpdateDescriptorSets(alloc_->device(),
        static_cast<uint32_t>(writes_.size()), writes_.data(),
        0, nullptr);
    return true;
}

} // namespace engine
