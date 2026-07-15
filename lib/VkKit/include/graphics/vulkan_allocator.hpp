#pragma once

#include <graphics/vulkan_context.hpp>

#include <vk_mem_alloc.h>

#include <cstdint>
#include <vector>

namespace vulkan_graphics {

struct HeapStatistics {
    uint32_t blockCount = 0;
    uint32_t allocationCount = 0;
    VkDeviceSize blockBytes = 0;
    VkDeviceSize allocationBytes = 0;
};

struct AllocatorStatistics {
    std::vector<HeapStatistics> heaps;
    VkDeviceSize totalBlockBytes = 0;
    VkDeviceSize totalAllocationBytes = 0;
};

struct MemoryBudget {
    uint32_t heapIndex = 0;
    VkDeviceSize allocationBytes = 0;
    VkDeviceSize usageBytes = 0;
    VkDeviceSize budgetBytes = 0;
};

class VulkanAllocator final {
  public:
    explicit VulkanAllocator(const VulkanContext& context);
    ~VulkanAllocator() noexcept;

    VulkanAllocator(const VulkanAllocator&) = delete;
    VulkanAllocator& operator=(const VulkanAllocator&) = delete;
    VulkanAllocator(VulkanAllocator&&) = delete;
    VulkanAllocator& operator=(VulkanAllocator&&) = delete;

    [[nodiscard]] VmaAllocator nativeHandle() const noexcept;
    [[nodiscard]] const VulkanContext& context() const noexcept;
    [[nodiscard]] AllocatorStatistics statistics() const;
    [[nodiscard]] std::vector<MemoryBudget> memoryBudgets() const;
    [[nodiscard]] bool isValid() const noexcept;

  private:
    const VulkanContext& context_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
};

} // namespace vulkan_graphics
