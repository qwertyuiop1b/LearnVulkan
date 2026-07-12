#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace vulkan_tutorial::production {

inline VkDeviceSize alignDown(VkDeviceSize value, VkDeviceSize alignment) {
    return value & ~(alignment - 1);
}

inline VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

inline VkMappedMemoryRange nonCoherentRange(VkDeviceMemory memory,
                                             VkDeviceSize offset,
                                             VkDeviceSize size,
                                             VkDeviceSize atomSize,
                                             VkDeviceSize allocationSize) {
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = memory;
    range.offset = alignDown(offset, atomSize);
    const VkDeviceSize requestedEnd = std::min(offset + size, allocationSize);
    const VkDeviceSize alignedEnd = std::min(alignUp(requestedEnd, atomSize), allocationSize);
    range.size = alignedEnd - range.offset;
    return range;
}

class FrameArena {
  public:
    explicit FrameArena(size_t capacity = 256 * 1024) : storage_(capacity) {}

    void reset() { cursor_ = 0; }

    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) {
        const size_t aligned = (cursor_ + alignment - 1) & ~(alignment - 1);
        if (aligned + size > storage_.size())
            throw std::bad_alloc();
        cursor_ = aligned + size;
        return storage_.data() + aligned;
    }

    template <typename T> T* allocate(size_t count = 1) {
        return static_cast<T*>(allocate(sizeof(T) * count, alignof(T)));
    }

    [[nodiscard]] size_t used() const { return cursor_; }
    [[nodiscard]] size_t capacity() const { return storage_.size(); }

  private:
    std::vector<std::byte> storage_;
    size_t cursor_ = 0;
};

class TimelineDeletionQueue {
  public:
    void retire(uint64_t timelineValue, std::function<void()> destroy) {
        entries_.push_back({timelineValue, std::move(destroy)});
    }

    size_t collect(uint64_t completedValue) {
        size_t count = 0;
        while (!entries_.empty() && entries_.front().value <= completedValue) {
            entries_.front().destroy();
            entries_.pop_front();
            ++count;
        }
        return count;
    }

    void drain() {
        while (!entries_.empty()) {
            entries_.front().destroy();
            entries_.pop_front();
        }
    }

    [[nodiscard]] size_t pending() const { return entries_.size(); }

  private:
    struct Entry {
        uint64_t value;
        std::function<void()> destroy;
    };
    std::deque<Entry> entries_;
};

struct FrameContext {
    uint32_t frameIndex = 0;
    uint64_t submitValue = 0;
    FrameArena arena;
    TimelineDeletionQueue deletions;

    FrameContext(size_t arenaCapacity = 256 * 1024) : arena(arenaCapacity) {}
};

struct CapabilityProfile {
    uint32_t apiVersion = VK_API_VERSION_1_0;
    std::string deviceName;
    bool timelineSemaphore = false;
    bool synchronization2 = false;
    bool dynamicRendering = false;
    bool bufferDeviceAddress = false;
    bool descriptorIndexing = false;
    bool meshShader = false;
    bool rayQuery = false;
    bool accelerationStructure = false;
    bool memoryBudget = false;
    bool presentId = false;
    bool presentWait = false;
    bool deviceFault = false;
    bool calibratedTimestamps = false;

    static CapabilityProfile query(VkPhysicalDevice physicalDevice) {
        CapabilityProfile profile;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        profile.apiVersion = properties.apiVersion;
        profile.deviceName = properties.deviceName;

        VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &features12;
        features12.pNext = &features13;
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
        profile.timelineSemaphore = features12.timelineSemaphore;
        profile.bufferDeviceAddress = features12.bufferDeviceAddress;
        profile.descriptorIndexing = features12.descriptorIndexing;
        profile.synchronization2 = features13.synchronization2;
        profile.dynamicRendering = features13.dynamicRendering;

        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data());
        std::unordered_set<std::string> names;
        for (const auto& extension : extensions)
            names.emplace(extension.extensionName);
        auto has = [&](const char* name) { return names.count(name) != 0; };
        profile.meshShader = has("VK_EXT_mesh_shader");
        profile.rayQuery = has("VK_KHR_ray_query");
        profile.accelerationStructure = has("VK_KHR_acceleration_structure");
        profile.memoryBudget = has("VK_EXT_memory_budget");
        profile.presentId = has("VK_KHR_present_id");
        profile.presentWait = has("VK_KHR_present_wait");
        profile.deviceFault = has("VK_EXT_device_fault");
        profile.calibratedTimestamps = has("VK_EXT_calibrated_timestamps");
        return profile;
    }
};

struct RunningStats {
    uint64_t samples = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double minimum = 1.0e30;
    double maximum = 0.0;

    void add(double value) {
        ++samples;
        const double delta = value - mean;
        mean += delta / static_cast<double>(samples);
        m2 += delta * (value - mean);
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }

    [[nodiscard]] double variance() const { return samples > 1 ? m2 / static_cast<double>(samples - 1) : 0.0; }
};

} // namespace vulkan_tutorial::production
