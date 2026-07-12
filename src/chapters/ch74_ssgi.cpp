/**
 * @file ch74_ssgi.cpp
 * @brief Screen-space indirect lighting from neighboring visible surfaces.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch74App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第74章：GPU SSGI"; }
    const char* fragmentShaderName() const override { return "advanced_ssgi.frag.spv"; }

    void configureChapter() override {
        interactive_.camera().setTarget({0.0f, -0.1f, -2.0f});
        interactive_.camera().setDistance(7.5f);
        interactive_.camera().setAngles(1.0f, 1.0f);
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("SSGI", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderFloat("Screen radius", &screenRadius_, 0.005f, 0.16f, "%.3f");
            ImGui::SliderInt("Samples", &sampleCount_, 1, 16);
            ImGui::SliderFloat("Bounce intensity", &intensity_, 0.0f, 3.0f, "%.2f");
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_, 0.0f};
        fillCameraFrame(push);
        push.values[5] = {screenRadius_, static_cast<float>(sampleCount_), intensity_, 0.0f};
    }

  private:
    float screenRadius_ = 0.085f;
    int sampleCount_ = 12;
    float intensity_ = 1.35f;
};

int main() {
    try {
        Ch74App app;
        app.run("ch74 - GPU SSGI", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch74 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
