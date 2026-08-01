#pragma once

#include <source_location>
#include <stdexcept>
#include <string>
#include <iostream>
#include <vulkan/vulkan_core.h>

namespace vk_engine::detail
{
inline void CheckVkResult(VkResult result, const char* expression, std::source_location location)
{
    if (result == VK_SUCCESS)
    {
        return;
    }

    throw std::runtime_error("Vulkan call failed: " + std::string(expression) + " returned VkResult " +
                             std::to_string(static_cast<int>(result)) + " at " + location.file_name() + ":" +
                             std::to_string(location.line()));
}
} // namespace vk_engine::detail

#define VK_CHECK(call)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        ::vk_engine::detail::CheckVkResult((call), #call, std::source_location::current());                            \
    } while (false)
