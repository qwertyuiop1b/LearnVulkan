/**
 * @file ch81_descriptor_buffer.cpp
 * @brief GPU material heap with descriptor-buffer capability detection.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <cstring>
#include <iostream>
#include <vector>

class Ch81App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第81章：GPU Descriptor Heap"; }
    const char* fragmentShaderName() const override { return "advanced_descriptor_heap.frag.spv"; }
    const char* computeShaderName() const override { return "advanced_descriptor_heap.comp.spv"; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        return {VkDeviceSize(MAX_MATERIALS) * sizeof(glm::vec4),
                VkDeviceSize(OBJECT_COUNT) * sizeof(uint32_t), 16, 16};
    }

    glm::uvec3 computeDispatch() const override {
        return {(std::max(MAX_MATERIALS, OBJECT_COUNT) + 63) / 64, 1, 1};
    }

    void onAdvancedInit() override {
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(physDev_, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physDev_, nullptr, &extensionCount, extensions.data());
        for (const auto& extension : extensions) {
            if (std::strcmp(extension.extensionName, "VK_EXT_descriptor_buffer") == 0) {
                descriptorBufferAvailable_ = true;
                break;
            }
        }
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Descriptor Heap", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderInt("Material count", &materialCount_, 8, static_cast<int>(MAX_MATERIALS));
            ImGui::SliderInt("Selected material", &selectedMaterial_, 0, materialCount_ - 1);
            ImGui::Text("VK_EXT_descriptor_buffer: %s", descriptorBufferAvailable_ ? "available" : "storage fallback");
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(materialCount_), static_cast<float>(OBJECT_COUNT),
                          static_cast<float>(selectedMaterial_), elapsed_};
        push.values[1] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), 0.0f, 0.0f};
    }

  private:
    static constexpr uint32_t MAX_MATERIALS = 256;
    static constexpr uint32_t OBJECT_COUNT = 32 * 18;
    int materialCount_ = 192;
    int selectedMaterial_ = 12;
    bool descriptorBufferAvailable_ = false;
};

int main() {
    try {
        Ch81App app;
        app.run("ch81 - GPU Descriptor Heap", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch81 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
