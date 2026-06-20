#pragma once
/**
 * @file descriptor_manager.hpp
 * @brief 第66章：描述符集管理
 *
 * 三个组件协同工作：
 *
 *   DescriptorLayoutCache
 *     哈希 {binding→type} → VkDescriptorSetLayout，避免重复创建
 *
 *   DescriptorAllocator
 *     管理多个 VkDescriptorPool，按需自动扩容（永不返回 VK_ERROR_OUT_OF_POOL_MEMORY）
 *
 *   DescriptorBuilder
 *     流式 API：.bind(n, buffer).bind(n, texture).build(set, layout)
 *     内部使用 DescriptorLayoutCache 和 DescriptorAllocator
 *
 * 使用示例（对比传统写法）：
 * @code
 *   // 传统写法（~30 行）
 *   VkDescriptorSetLayoutBinding b0{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ...};
 *   VkDescriptorSetLayoutCreateInfo lci{...};
 *   VK_CHECK(vkCreateDescriptorSetLayout(...));
 *   VkDescriptorSetAllocateInfo ai{...};
 *   VK_CHECK(vkAllocateDescriptorSets(...));
 *   VkWriteDescriptorSet w0{...};
 *   vkUpdateDescriptorSets(...);
 *
 *   // DescriptorBuilder（~4 行）
 *   VkDescriptorSet set;
 *   VkDescriptorSetLayout layout;
 *   DescriptorBuilder(allocator, layoutCache)
 *       .bindBuffer(0, cameraUBO.descriptorInfo(fi))
 *       .bindImage(1, albedoTex.descriptorInfo())
 *       .build(set, layout);
 * @endcode
 */

#include "rhi_device.hpp"
#include <cstring>
#include <unordered_map>
#include <vector>

namespace engine {

// ─── Layout 缓存 ──────────────────────────────────────────────────────────

class DescriptorLayoutCache {
  public:
    void init(RHIDevice& dev) {
        dev_ = &dev;
    }
    void destroy();

    /// 获取或创建 DescriptorSetLayout（自动去重）
    [[nodiscard]] VkDescriptorSetLayout getOrCreate(const std::vector<VkDescriptorSetLayoutBinding>& bindings);

    [[nodiscard]] size_t cachedCount() const {
        return cache_.size();
    }

  private:
    struct LayoutKey {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bool operator==(const LayoutKey& o) const;
    };
    struct LayoutKeyHash {
        size_t operator()(const LayoutKey& k) const;
    };

    RHIDevice* dev_ = nullptr;
    std::unordered_map<LayoutKey, VkDescriptorSetLayout, LayoutKeyHash> cache_;
};

// ─── 描述符分配器 ─────────────────────────────────────────────────────────

/**
 * @brief 自动扩容的 Descriptor Pool 管理器
 *
 * 传统做法：手动计算需要多少个 UBO、多少个 Sampler，一次性创建一个大 Pool。
 * 问题：难以估计，经常不够或浪费。
 *
 * DescriptorAllocator：
 *   - 维护一个"当前 pool"和一个"满了"的旧 pool 列表
 *   - 分配失败时自动创建新 pool，继续分配
 *   - reset() 归还所有 set（用于每帧重建描述符的场景）
 *   - 或者 freeSet() 按需释放单个 set
 */
class DescriptorAllocator {
  public:
    void init(RHIDevice& dev);
    void destroy();
    void reset(); ///< 归还所有描述符集（但不销毁 pool）

    [[nodiscard]] VkDescriptorSet allocate(VkDescriptorSetLayout layout);

    [[nodiscard]] size_t poolCount() const {
        return allPools_.size();
    }
    [[nodiscard]] size_t totalAllocated() const {
        return totalAllocated_;
    }
    [[nodiscard]] RHIDevice* dev() const {
        return dev_;
    }
    [[nodiscard]] VkDevice device() const {
        return dev_ ? dev_->device() : VK_NULL_HANDLE;
    }

  private:
    [[nodiscard]] VkDescriptorPool createPool();
    [[nodiscard]] VkDescriptorPool grabPool();

    RHIDevice* dev_ = nullptr;
    VkDescriptorPool currentPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> usedPools_;
    std::vector<VkDescriptorPool> freePools_;
    std::vector<VkDescriptorPool> allPools_;
    size_t totalAllocated_ = 0;
};

// ─── 描述符构建器 ─────────────────────────────────────────────────────────

/**
 * @brief 流式描述符集构建器
 *
 * 设计：每次调用 build() 生成一个新的 set，不可复用（请为每帧创建）。
 * 使用 DescriptorLayoutCache 使 layout 在相同绑定时复用。
 */
class DescriptorBuilder {
  public:
    DescriptorBuilder(DescriptorAllocator& alloc, DescriptorLayoutCache& cache) : alloc_(&alloc), cache_(&cache) {}

    // ── 绑定 Buffer ──────────────────────────────────────────────────────
    DescriptorBuilder& bindBuffer(uint32_t binding,
                                  const VkDescriptorBufferInfo& info,
                                  VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                  VkShaderStageFlags stages = VK_SHADER_STAGE_ALL_GRAPHICS);

    // ── 绑定 Image / Sampler ─────────────────────────────────────────────
    DescriptorBuilder& bindImage(uint32_t binding,
                                 const VkDescriptorImageInfo& info,
                                 VkDescriptorType type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 VkShaderStageFlags stages = VK_SHADER_STAGE_FRAGMENT_BIT);

    // ── 绑定 StorageBuffer ───────────────────────────────────────────────
    DescriptorBuilder& bindStorageBuffer(uint32_t binding,
                                         const VkDescriptorBufferInfo& info,
                                         VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT);

    // ── 绑定 StorageImage ─────────────────────────────────────────────────
    DescriptorBuilder& bindStorageImage(uint32_t binding,
                                        const VkDescriptorImageInfo& info,
                                        VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT);

    /// 分配 set 并写入绑定，返回 false 表示失败
    bool build(VkDescriptorSet& outSet, VkDescriptorSetLayout& outLayout);
    /// 仅写入（不分配），set 由外部预先分配
    bool write(VkDescriptorSet set);

  private:
    DescriptorAllocator* alloc_;
    DescriptorLayoutCache* cache_;

    std::vector<VkDescriptorSetLayoutBinding> bindings_;
    std::vector<VkWriteDescriptorSet> writes_;
    std::vector<VkDescriptorBufferInfo> bufInfos_;
    std::vector<VkDescriptorImageInfo> imgInfos_;
};

} // namespace engine
