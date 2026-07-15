#include <graphics/vulkan_allocator.hpp>

#include <array>
#include <stdexcept>

namespace vulkan_graphics {
namespace {

HeapStatistics toHeapStatistics(const VmaStatistics& statistics) {
    return {
        statistics.blockCount,
        statistics.allocationCount,
        statistics.blockBytes,
        statistics.allocationBytes,
    };
}

} // namespace

VulkanAllocator::VulkanAllocator(const VulkanContext& context) : context_(context) {
    if (static_cast<VkDevice>(context.device()) == VK_NULL_HANDLE)
        throw std::invalid_argument("VulkanAllocator requires an initialized VulkanContext");

    VmaAllocatorCreateInfo createInfo{};
    createInfo.instance = static_cast<VkInstance>(context.instance());
    createInfo.physicalDevice = static_cast<VkPhysicalDevice>(context.physicalDevice());
    createInfo.device = static_cast<VkDevice>(context.device());
    createInfo.vulkanApiVersion = context.apiVersion();

    const VkResult result = vmaCreateAllocator(&createInfo, &allocator_);
    if (result != VK_SUCCESS)
        throw std::runtime_error("Failed to create Vulkan Memory Allocator");
}

VulkanAllocator::~VulkanAllocator() noexcept {
    if (allocator_ != VK_NULL_HANDLE)
        vmaDestroyAllocator(allocator_);
}

VmaAllocator VulkanAllocator::nativeHandle() const noexcept {
    return allocator_;
}

const VulkanContext& VulkanAllocator::context() const noexcept {
    return context_;
}

AllocatorStatistics VulkanAllocator::statistics() const {
    VmaTotalStatistics vmaStatistics{};
    vmaCalculateStatistics(allocator_, &vmaStatistics);

    AllocatorStatistics result{};
    const uint32_t heapCount = context_.memoryProperties().memoryHeapCount;
    result.heaps.reserve(heapCount);
    for (uint32_t heapIndex = 0; heapIndex < heapCount; ++heapIndex)
        result.heaps.push_back(toHeapStatistics(vmaStatistics.memoryHeap[heapIndex].statistics));

    result.totalBlockBytes = vmaStatistics.total.statistics.blockBytes;
    result.totalAllocationBytes = vmaStatistics.total.statistics.allocationBytes;
    return result;
}

std::vector<MemoryBudget> VulkanAllocator::memoryBudgets() const {
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> vmaBudgets{};
    vmaGetHeapBudgets(allocator_, vmaBudgets.data());

    const uint32_t heapCount = context_.memoryProperties().memoryHeapCount;
    std::vector<MemoryBudget> result;
    result.reserve(heapCount);
    for (uint32_t heapIndex = 0; heapIndex < heapCount; ++heapIndex) {
        const VmaBudget& budget = vmaBudgets[heapIndex];
        result.push_back({
            heapIndex,
            budget.statistics.allocationBytes,
            budget.usage,
            budget.budget,
        });
    }
    return result;
}

bool VulkanAllocator::isValid() const noexcept {
    return allocator_ != VK_NULL_HANDLE;
}

} // namespace vulkan_graphics
