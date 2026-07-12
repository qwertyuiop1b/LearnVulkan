/**
 * @file ch71_depth_of_field.cpp
 * @brief Multi-aperture GPU depth of field on an analytic 3D scene.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch71App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第71章：GPU 景深"; }
    const char* fragmentShaderName() const override { return "advanced_dof.frag.spv"; }

    void configureChapter() override {
        bgColor_ = {0.015f, 0.025f, 0.05f};
        interactive_.camera().setTarget({0.0f, -0.05f, -1.4f});
        interactive_.camera().setDistance(7.8f);
        interactive_.camera().setAngles(8.0f, 3.0f);
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Depth of Field", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderFloat("Focus distance", &focusDistance_, 2.0f, 14.0f, "%.2f m");
            ImGui::SliderFloat("Aperture", &aperture_, 0.0f, 0.28f, "%.3f");
            ImGui::SliderInt("Aperture samples", &sampleCount_, 1, 12);
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_, 0.0f};
        fillCameraFrame(push);
        push.values[5] = {focusDistance_, aperture_, static_cast<float>(sampleCount_), 0.0f};
    }

  private:
    float focusDistance_ = 7.2f;
    float aperture_ = 0.105f;
    int sampleCount_ = 8;
};

int main() {
    try {
        Ch71App app;
        app.run("ch71 - GPU Depth of Field", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch71 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
