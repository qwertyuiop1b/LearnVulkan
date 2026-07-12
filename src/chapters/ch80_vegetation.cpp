/**
 * @file ch80_vegetation.cpp
 * @brief Compute-scattered vegetation rendered with an indirect draw.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch80App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第80章：GPU 植被"; }
    const char* vertexShaderName() const override { return "advanced_vegetation.vert.spv"; }
    const char* fragmentShaderName() const override { return "advanced_vegetation.frag.spv"; }
    const char* computeShaderName() const override { return "advanced_vegetation.comp.spv"; }
    bool chapterNeedsDepth() const override { return true; }
    int32_t indirectDrawBufferIndex() const override { return 1; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        return {VkDeviceSize(MAX_INSTANCES) * sizeof(glm::vec4) * 2,
                sizeof(VkDrawIndirectCommand), 16, 16};
    }

    std::array<VkBufferUsageFlags, STORAGE_BUFFER_COUNT> extraBufferUsage() const override {
        return {0, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, 0, 0};
    }

    glm::uvec3 computeDispatch() const override { return {(MAX_INSTANCES + 63) / 64, 1, 1}; }

    void configureChapter() override {
        bgColor_ = {0.42f, 0.57f, 0.68f};
        interactive_.camera().setTarget({0.0f, 1.2f, 0.0f});
        interactive_.camera().setDistance(23.0f);
        interactive_.camera().setAngles(35.0f, 17.0f);
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("GPU Vegetation", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderFloat("Density", &density_, 0.05f, 1.0f, "%.2f");
            ImGui::SliderFloat("Field size", &fieldSize_, 10.0f, 48.0f, "%.1f m");
            ImGui::SliderFloat("Wind", &wind_, 0.0f, 1.4f, "%.2f");
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        const float aspect = static_cast<float>(extent_.width) / static_cast<float>(std::max(extent_.height, 1u));
        const glm::mat4 viewProjection = interactive_.camera().projectionMatrix(aspect, 55.0f, 0.1f, 180.0f) *
                                         interactive_.camera().viewMatrix();
        for (uint32_t column = 0; column < 4; ++column)
            push.values[column] = viewProjection[column];
        push.values[4] = glm::vec4(interactive_.camera().eyePosition(), elapsed_);
        push.values[5] = {fieldSize_, density_, static_cast<float>(MAX_INSTANCES), wind_};
    }

  private:
    static constexpr uint32_t MAX_INSTANCES = 6400;
    float density_ = 0.72f;
    float fieldSize_ = 32.0f;
    float wind_ = 0.62f;
};

int main() {
    try {
        Ch80App app;
        app.run("ch80 - GPU Vegetation", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch80 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
