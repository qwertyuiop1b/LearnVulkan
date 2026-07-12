/**
 * @file ch79_terrain_lod.cpp
 * @brief Distance-adaptive procedural terrain sampling and LOD visualization.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch79App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第79章：GPU 地形 LOD"; }
    const char* fragmentShaderName() const override { return "advanced_terrain_lod.frag.spv"; }

    void configureChapter() override {
        interactive_.camera().setTarget({0.0f, 0.0f, -12.0f});
        interactive_.camera().setDistance(30.0f);
        interactive_.camera().setAngles(24.0f, 18.0f);
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Terrain LOD", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderFloat("Height scale", &heightScale_, 1.0f, 14.0f, "%.1f m");
            ImGui::SliderFloat("LOD bias", &lodBias_, -2.0f, 3.0f, "%.1f");
            ImGui::Checkbox("Show LOD cells", &showLod_);
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_, 0.0f};
        fillCameraFrame(push);
        push.values[5] = {heightScale_, lodBias_, showLod_ ? 1.0f : 0.0f, 0.0f};
    }

  private:
    float heightScale_ = 7.5f;
    float lodBias_ = 0.0f;
    bool showLod_ = true;
};

int main() {
    try {
        Ch79App app;
        app.run("ch79 - GPU Terrain LOD", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch79 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
