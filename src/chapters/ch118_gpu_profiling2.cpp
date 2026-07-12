/** @file ch118_gpu_profiling2.cpp @brief Timestamp queries, calibrated capability and frame statistics. */

#include <vulkan_tutorial/engine/engineering_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch118App final : public EngineeringGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第118章：GPU Profiling 2.0"; }
    uint32_t engineeringMode() const override { return 118; }

    void onEngineeringInit() override {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physDev_, &properties);
        timestampPeriod_ = properties.limits.timestampPeriod;
        queryAvailable_ = properties.limits.timestampComputeAndGraphics;
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = QUERY_COUNT;
        VK_CHECK(vkCreateQueryPool(device_, &info, nullptr, &queryPool_));
    }

    void onEngineeringShutdown() override { vkDestroyQueryPool(device_, queryPool_, nullptr); }

    void beforeAdvancedCompute(VkCommandBuffer command, uint32_t frame) override {
        if (!queryAvailable_) return;
        const uint32_t base = (frame % 2) * 2;
        vkCmdResetQueryPool(command, queryPool_, base, 2);
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, base);
    }

    void afterAdvancedDraw(VkCommandBuffer command, uint32_t frame) override {
        if (!queryAvailable_) return;
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, (frame % 2) * 2 + 1);
    }

    void updateChapter() override {
        if (!queryAvailable_ || frameNumber_ < 3) return;
        const uint32_t frame = static_cast<uint32_t>((frameNumber_ + 1) % 2);
        uint64_t values[2]{};
        if (vkGetQueryPoolResults(device_, queryPool_, frame * 2, 2, sizeof(values), values,
                                  sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && values[1] >= values[0]) {
            lastGpuMs_ = static_cast<double>(values[1] - values[0]) * timestampPeriod_ / 1.0e6;
            stats_.add(lastGpuMs_);
        }
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("GPU Profiling", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            drawCapabilitySummary();
            ImGui::Text("Timestamp query: %s", queryAvailable_ ? "available" : "unsupported");
            ImGui::Text("Timestamp period: %.3f ns", timestampPeriod_);
            ImGui::Text("Last GPU interval: %.3f ms", lastGpuMs_);
            ImGui::Text("Mean/min/max: %.3f / %.3f / %.3f ms", stats_.mean, stats_.minimum, stats_.maximum);
            ImGui::Text("Calibrated timestamps: %s", profile_.calibratedTimestamps ? "available" : "fallback");
        }
        ImGui::End();
    }

    glm::vec4 engineeringParameters() const override {
        return {static_cast<float>(lastGpuMs_), static_cast<float>(stats_.mean),
                static_cast<float>(timestampPeriod_), queryAvailable_ ? 1.0f : 0.0f};
    }

  private:
    static constexpr uint32_t QUERY_COUNT = 4;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    float timestampPeriod_ = 0.0f;
    bool queryAvailable_ = false;
    double lastGpuMs_ = 0.0;
    vulkan_tutorial::production::RunningStats stats_;
};

int main() {
    try { Ch118App app; app.run("ch118 - GPU Profiling 2.0", 1280, 720); }
    catch (const std::exception& error) { std::cerr << "ch118 failed: " << error.what() << '\n'; return 1; }
    return 0;
}
