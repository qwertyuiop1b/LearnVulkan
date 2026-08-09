#pragma once

#include "vk_context.h"

#include <cstdint>

#include <vulkan/vulkan_raii.hpp>

struct GLFWwindow;
struct ImGuiContext;

namespace vk_engine
{
/**
 * @brief RAII wrapper around Dear ImGui, integrating the GLFW platform backend
 *        and the Vulkan renderer backend using dynamic rendering.
 *
 * UI is rendered onto the offscreen draw image inside the render callback, so it
 * blends with the scene in the same frame. Call BeginFrame() in the main loop and
 * Render(cmd) between beginRendering()/endRendering() in the draw callback.
 */
class VkImGui
{
public:
    VkImGui(const VkContext& inContext,
            GLFWwindow* inWindow,
            vk::Format drawImageFormat,
            uint32_t minImageCount,
            uint32_t imageCount);
    ~VkImGui();

    VkImGui(const VkImGui&) = delete;
    VkImGui& operator=(const VkImGui&) = delete;
    VkImGui(VkImGui&&) = delete;
    VkImGui& operator=(VkImGui&&) = delete;

    void BeginFrame();
    void Render(vk::CommandBuffer commandBuffer);

private:
    ImGuiContext* imguiContext{nullptr};
};
} // namespace vk_engine
