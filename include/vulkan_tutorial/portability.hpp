#pragma once

/**
 * @file portability.hpp
 * @brief 跨平台可移植性辅助：macOS + MoltenVK 所需的实例标志
 *
 * 所有章节共用，避免在每处创建 VkInstance 时重复书写 #ifdef __APPLE__。
 * 该头文件仅依赖 <vulkan/vulkan.h>，对 headless 目标（ch115）也安全。
 */

#include <vulkan/vulkan.h>

/// macOS + MoltenVK：允许枚举非完全符合 Vulkan 规范的物理设备
/// （如 MoltenVK 暴露的 portability 子集设备）。
/// 非 Apple 平台为空操作，调用方无需任何条件编译。
inline void enablePortabilityBit(VkInstanceCreateInfo& createInfo) {
#ifdef __APPLE__
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
}
