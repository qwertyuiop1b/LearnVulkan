/**
 * @file ch82_device_generated_commands.cpp
 * @brief Compute-generated object data and indirect draw command with DGC detection.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <cstring>
#include <iostream>
#include <vector>

class Ch82App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第82章：GPU 生成绘制命令"; }
    const char* vertexShaderName() const override { return "advanced_dgc.vert.spv"; }
    const char* fragmentShaderName() const override { return "advanced_dgc.frag.spv"; }
    const char* computeShaderName() const override { return "advanced_dgc.comp.spv"; }
    bool chapterUsesAlphaBlend() const override { return true; }
    int32_t indirectDrawBufferIndex() const override { return 1; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        return {VkDeviceSize(MAX_OBJECTS) * sizeof(glm::vec4) * 2,
                sizeof(VkDrawIndirectCommand), 16, 16};
    }

    std::array<VkBufferUsageFlags, STORAGE_BUFFER_COUNT> extraBufferUsage() const override {
        return {0, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, 0, 0};
    }

    glm::uvec3 computeDispatch() const override { return {(MAX_OBJECTS + 63) / 64, 1, 1}; }

    void configureChapter() override {
        bgColor_ = {0.008f, 0.014f, 0.03f};
        interactive_.camera().setTarget({0.0f, 0.0f, -5.0f});
        interactive_.camera().setDistance(13.0f);
        interactive_.camera().setAngles(0.0f, 0.0f);
    }

    void onAdvancedInit() override {
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(physDev_, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physDev_, nullptr, &extensionCount, extensions.data());
        for (const auto& extension : extensions) {
            if (std::strcmp(extension.extensionName, "VK_EXT_device_generated_commands") == 0) {
                dgcAvailable_ = true;
                break;
            }
        }
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("GPU Generated Commands", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderInt("Object count", &objectCount_, 128, static_cast<int>(MAX_OBJECTS));
            ImGui::Text("VK_EXT_device_generated_commands: %s", dgcAvailable_ ? "available" : "indirect fallback");
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        const float aspect = static_cast<float>(extent_.width) / static_cast<float>(std::max(extent_.height, 1u));
        const glm::mat4 viewProjection = interactive_.camera().projectionMatrix(aspect, 57.0f, 0.1f, 120.0f) *
                                         interactive_.camera().viewMatrix();
        for (uint32_t column = 0; column < 4; ++column)
            push.values[column] = viewProjection[column];
        push.values[4] = glm::vec4(interactive_.camera().eyePosition(), elapsed_);
        push.values[5] = {static_cast<float>(objectCount_), 0.0f, 0.0f, 0.0f};
    }

  private:
    static constexpr uint32_t MAX_OBJECTS = 4096;
    int objectCount_ = 2304;
    bool dgcAvailable_ = false;
};

int main() {
    try {
        Ch82App app;
        app.run("ch82 - GPU Generated Commands", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch82 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
