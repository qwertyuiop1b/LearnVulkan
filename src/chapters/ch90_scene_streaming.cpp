/**
 * @file ch90_scene_streaming.cpp
 * @brief GPU chunk residency map consumed directly by terrain rendering.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch90App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第90章：GPU 场景驻留与 Chunk 渲染"; }
    const char* fragmentShaderName() const override { return "engine_streaming.frag.spv"; }
    const char* computeShaderName() const override { return "engine_streaming.comp.spv"; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        return {GRID * GRID * sizeof(glm::vec4), 16, 16, 16};
    }

    glm::uvec3 computeDispatch() const override { return {(GRID * GRID + 63) / 64, 1, 1}; }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("GPU Chunk Residency", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderInt("Resident radius", &residentRadius_, 2, 7);
            ImGui::SliderFloat("Travel speed", &travelSpeed_, 0.0f, 1.5f);
            ImGui::Text("Chunk table: %u x %u", GRID, GRID);
            ImGui::TextUnformatted("Compute updates residency; fragment pass renders resident terrain.");
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_,
                          static_cast<float>(residentRadius_)};
        push.values[1] = {static_cast<float>(GRID), travelSpeed_, 0.0f, 0.0f};
        fillCameraFrame(push, 2, 54.0f);
    }

    void configureChapter() override {
        bgColor_ = {0.015f, 0.025f, 0.04f};
        interactive_.camera().setTarget({0.0f, 0.0f, -5.0f});
        interactive_.camera().setDistance(12.0f);
        interactive_.camera().setAngles(0.0f, -0.32f);
    }

  private:
    static constexpr uint32_t GRID = 16;
    int residentRadius_ = 5;
    float travelSpeed_ = 0.45f;
};

int main() {
    try {
        Ch90App app;
        app.run("ch90 - GPU Scene Residency", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch90 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
