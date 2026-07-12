/**
 * @file ch110_frame_context.cpp
 * @brief Per-frame linear arenas and GPU-timeline deferred destruction.
 */

#include <vulkan_tutorial/engine/engineering_gpu_chapter.hpp>

#include <imgui.h>
#include <array>
#include <iostream>

class Ch110App final : public EngineeringGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第110章：FrameContext"; }
    uint32_t engineeringMode() const override { return 110; }

    void onEngineeringInit() override {
        profile_ = vulkan_tutorial::production::CapabilityProfile::query(physDev_);
        if (!profile_.timelineSemaphore)
            return;
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        VK_CHECK(vkCreateSemaphore(device_, &info, nullptr, &timeline_));
    }

    void updateChapter() override {
        const uint32_t index = static_cast<uint32_t>(frameNumber_ % contexts_.size());
        auto& context = contexts_[index];
        context.frameIndex = index;
        context.arena.reset();
        const size_t allocationCount = 12 + static_cast<size_t>(frameNumber_ % 20);
        for (size_t i = 0; i < allocationCount; ++i)
            std::memset(context.arena.allocate(192 + i * 13, 64), int(i), 192 + i * 13);

        if (timeline_ == VK_NULL_HANDLE)
            return;
        const uint64_t signalValue = ++submittedValue_;
        VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signal.semaphore = timeline_;
        signal.value = signalValue;
        signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.signalSemaphoreInfoCount = 1;
        submit.pSignalSemaphoreInfos = &signal;
        VK_CHECK(vkQueueSubmit2(gQueue_, 1, &submit, VK_NULL_HANDLE));
        context.submitValue = signalValue;
        context.deletions.retire(signalValue + contexts_.size(), [this] { ++destroyedObjects_; });

        vkGetSemaphoreCounterValue(device_, timeline_, &completedValue_);
        for (auto& frame : contexts_)
            collectedObjects_ += frame.deletions.collect(completedValue_);
        activeArenaBytes_ = context.arena.used();
    }

    void onEngineeringShutdown() override {
        if (timeline_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
            for (auto& context : contexts_)
                context.deletions.drain();
            vkDestroySemaphore(device_, timeline_, nullptr);
        }
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Frame Resources", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Frame contexts: %zu", contexts_.size());
            ImGui::Text("Arena used: %.1f / 256 KiB", activeArenaBytes_ / 1024.0f);
            ImGui::Text("Timeline submitted/completed: %llu / %llu",
                        static_cast<unsigned long long>(submittedValue_),
                        static_cast<unsigned long long>(completedValue_));
            size_t pending = 0;
            for (const auto& context : contexts_)
                pending += context.deletions.pending();
            ImGui::Text("Deletion queue pending: %zu", pending);
            ImGui::Text("Collected callbacks: %llu", static_cast<unsigned long long>(collectedObjects_));
        }
        ImGui::End();
    }

    glm::vec4 engineeringParameters() const override {
        return {static_cast<float>(activeArenaBytes_) / (256.0f * 1024.0f),
                static_cast<float>(submittedValue_ - completedValue_),
                static_cast<float>(collectedObjects_ % 100), 0.0f};
    }

  private:
    vulkan_tutorial::production::CapabilityProfile profile_{};
    std::array<vulkan_tutorial::production::FrameContext, 3> contexts_{};
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    uint64_t submittedValue_ = 0;
    uint64_t completedValue_ = 0;
    uint64_t destroyedObjects_ = 0;
    uint64_t collectedObjects_ = 0;
    size_t activeArenaBytes_ = 0;
};

int main() {
    try {
        Ch110App app;
        app.run("ch110 - FrameContext", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch110 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
