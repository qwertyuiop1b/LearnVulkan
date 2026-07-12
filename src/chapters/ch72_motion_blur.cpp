/**
 * @file ch72_motion_blur.cpp
 * @brief Temporal supersampling motion blur for moving 3D objects.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch72App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第72章：GPU 运动模糊"; }
    const char* fragmentShaderName() const override { return "advanced_motion_blur.frag.spv"; }

    void configureChapter() override {
        bgColor_ = {0.01f, 0.02f, 0.045f};
        interactive_.camera().setTarget({0.0f, -0.1f, -2.0f});
        interactive_.camera().setDistance(8.2f);
        interactive_.camera().setAngles(2.0f, 2.0f);
    }

    void updateChapter() override {
        if (!animate_)
            elapsed_ -= 0.016f;
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Motion Blur", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Checkbox("Animate", &animate_);
            ImGui::SliderFloat("Shutter interval", &shutter_, 0.0f, 1.5f, "%.2f s");
            ImGui::SliderInt("Temporal samples", &sampleCount_, 1, 16);
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_, 0.0f};
        fillCameraFrame(push);
        push.values[5] = {shutter_, static_cast<float>(sampleCount_), 0.0f, 0.0f};
    }

  private:
    float shutter_ = 0.68f;
    int sampleCount_ = 10;
    bool animate_ = true;
};

int main() {
    try {
        Ch72App app;
        app.run("ch72 - GPU Motion Blur", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch72 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
