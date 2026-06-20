#pragma once

/**
 * @file interactive_chapter.hpp
 * @brief ch38+ 交互工具：轨道相机 + ImGui 调试面板 + GPU/CPU 性能统计
 */

#include <vulkan_tutorial/camera_controller.hpp>
#include <vulkan_tutorial/frame_profiler.hpp>

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <string>

namespace vulkan_tutorial {

struct InteractiveInitInfo {
    GLFWwindow* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
    uint32_t imageCount = 2;
    uint32_t maxFramesInFlight = 2;
};

class InteractiveChapterTools {
  public:
    void attachInput(GLFWwindow* window);
    bool initVulkan(const InteractiveInitInfo& info);
    void shutdown(VkDevice device);
    void onSwapchainRecreated(VkRenderPass renderPass, VkFormat format, uint32_t imageCount);
    void beginFrame(float deltaSeconds);
    void buildDebugPanel(const char* chapterTitle);
    void beginGpuSection(VkCommandBuffer cmd, uint32_t frameIndex);
    void endGpuSection(VkCommandBuffer cmd, uint32_t frameIndex);
    void renderUi(VkCommandBuffer cmd);
    void endFrame(uint32_t frameIndex);
    void updateWindowTitle() const;
    OrbitCamera& camera() {
        return camera_;
    }
    const OrbitCamera& camera() const {
        return camera_;
    }
    const FrameStats& stats() const {
        return stats_;
    }
    bool wantsCaptureMouse() const;
    bool wantsCaptureKeyboard() const;
    bool hasUi() const {
        return imguiReady_;
    }

  private:
    InteractiveInitInfo info_{};
    OrbitCamera camera_{};
    FrameTimer frameTimer_{};
    GpuTimestampProfiler gpuProfiler_{};
    FrameStats stats_{};
    bool initialized_ = false;
    bool imguiReady_ = false;
    VkDescriptorPool imguiPool_ = VK_NULL_HANDLE;
    std::string windowTitleBase_;
    float lastDeltaSeconds_ = 0.016f;
};

} // namespace vulkan_tutorial
