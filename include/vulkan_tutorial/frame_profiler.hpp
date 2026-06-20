#pragma once

/**
 * @file frame_profiler.hpp
 * @brief CPU 帧率统计 + GPU 时间戳查询
 */

#include <vulkan_tutorial/utils.hpp>

#include <chrono>
#include <string>

namespace vulkan_tutorial {

struct FrameStats {
    float cpuFrameMs = 0.0f;
    float cpuFps = 0.0f;
    float gpuFrameMs = 0.0f;
    bool gpuTimingAvailable = false;
};

class FrameTimer {
  public:
    void beginFrame() {
        frameStart_ = std::chrono::steady_clock::now();
    }
    void endFrame() {
        const auto now = std::chrono::steady_clock::now();
        cpuFrameMs_ = std::chrono::duration<float, std::milli>(now - frameStart_).count();
        ++frameCounter_;
        const float elapsed = std::chrono::duration<float>(now - fpsStart_).count();
        if (elapsed >= 0.5f) {
            cpuFps_ = static_cast<float>(frameCounter_) / elapsed;
            frameCounter_ = 0;
            fpsStart_ = now;
        }
    }
    float cpuFrameMs() const {
        return cpuFrameMs_;
    }
    float cpuFps() const {
        return cpuFps_;
    }

  private:
    std::chrono::steady_clock::time_point frameStart_{};
    std::chrono::steady_clock::time_point fpsStart_{std::chrono::steady_clock::now()};
    uint32_t frameCounter_ = 0;
    float cpuFrameMs_ = 0.0f;
    float cpuFps_ = 0.0f;
};

class GpuTimestampProfiler {
  public:
    bool init(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t maxFramesInFlight) {
        maxFrames_ = maxFramesInFlight;
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        timestampPeriodNs_ = props.limits.timestampPeriod;
        if (props.limits.timestampComputeAndGraphics == VK_FALSE) {
            available_ = false;
            return false;
        }
        VkQueryPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        poolInfo.queryCount = maxFrames_ * 2;
        if (vkCreateQueryPool(device, &poolInfo, nullptr, &queryPool_) != VK_SUCCESS) {
            available_ = false;
            return false;
        }
        available_ = true;
        return true;
    }
    void shutdown(VkDevice device) {
        if (queryPool_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, queryPool_, nullptr);
            queryPool_ = VK_NULL_HANDLE;
        }
    }
    bool isAvailable() const {
        return available_;
    }
    void reset(VkCommandBuffer cmd, uint32_t frameIndex) {
        if (!available_)
            return;
        vkCmdResetQueryPool(cmd, queryPool_, frameIndex * 2, 2);
    }
    void writeStart(VkCommandBuffer cmd, uint32_t frameIndex) {
        if (!available_)
            return;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, frameIndex * 2);
    }
    void writeEnd(VkCommandBuffer cmd, uint32_t frameIndex) {
        if (!available_)
            return;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, frameIndex * 2 + 1);
    }
    float readGpuFrameMs(VkDevice device, uint32_t frameIndex) {
        if (!available_)
            return 0.0f;
        uint64_t timestamps[2] = {};
        const VkResult result = vkGetQueryPoolResults(device,
                                                      queryPool_,
                                                      frameIndex * 2,
                                                      2,
                                                      sizeof(timestamps),
                                                      timestamps,
                                                      sizeof(uint64_t),
                                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (result != VK_SUCCESS)
            return 0.0f;
        const float delta = static_cast<float>(timestamps[1] - timestamps[0]) * timestampPeriodNs_;
        return delta / 1.0e6f;
    }

  private:
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    float timestampPeriodNs_ = 1.0f;
    uint32_t maxFrames_ = 2;
    bool available_ = false;
};

inline std::string formatFrameStats(const FrameStats& stats) {
    if (stats.gpuTimingAvailable)
        return "FPS " + std::to_string(static_cast<int>(stats.cpuFps + 0.5f)) + " | CPU " +
               std::to_string(static_cast<int>(stats.cpuFrameMs + 0.5f)) + " ms" + " | GPU " +
               std::to_string(static_cast<int>(stats.gpuFrameMs * 100.0f + 0.5f) / 100.0f) + " ms";
    return "FPS " + std::to_string(static_cast<int>(stats.cpuFps + 0.5f)) + " | CPU " +
           std::to_string(static_cast<int>(stats.cpuFrameMs + 0.5f)) + " ms";
}

} // namespace vulkan_tutorial
