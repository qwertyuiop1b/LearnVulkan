/**
 * @file ch78_async_compute.cpp
 * @brief Compute-produced simulation field consumed by a graphics pass.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>
#include <vector>

class Ch78App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第78章：GPU Compute/Graphics 调度"; }
    const char* fragmentShaderName() const override { return "advanced_async_field.frag.spv"; }
    const char* computeShaderName() const override { return "advanced_async_field.comp.spv"; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        return {VkDeviceSize(FIELD_WIDTH) * FIELD_HEIGHT * sizeof(glm::vec4), 16, 16, 16};
    }

    glm::uvec3 computeDispatch() const override {
        return {(FIELD_WIDTH + 7) / 8, (FIELD_HEIGHT + 7) / 8, 1};
    }

    void onAdvancedInit() override {
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physDev_, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> properties(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physDev_, &familyCount, properties.data());
        for (const auto& family : properties) {
            if ((family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 &&
                (family.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
                hasDedicatedComputeFamily_ = true;
                break;
            }
        }
    }

    void updateChapter() override {
        if (!animate_)
            elapsed_ -= 0.016f;
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Compute Scheduling", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Checkbox("Animate simulation", &animate_);
            ImGui::Text("Dedicated compute family: %s", hasDedicatedComputeFamily_ ? "available" : "fallback");
            ImGui::Text("Field: %u x %u", FIELD_WIDTH, FIELD_HEIGHT);
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_, 0.0f};
        push.values[1] = {static_cast<float>(FIELD_WIDTH), static_cast<float>(FIELD_HEIGHT), 0.0f, 0.0f};
    }

  private:
    static constexpr uint32_t FIELD_WIDTH = 320;
    static constexpr uint32_t FIELD_HEIGHT = 180;
    bool animate_ = true;
    bool hasDedicatedComputeFamily_ = false;
};

int main() {
    try {
        Ch78App app;
        app.run("ch78 - GPU Compute Graphics Scheduling", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch78 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
