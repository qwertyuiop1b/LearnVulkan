/**
 * @file ch73_atmospheric_scattering.cpp
 * @brief Rayleigh and Mie volumetric atmosphere integration.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <cmath>
#include <iostream>

class Ch73App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第73章：GPU 大气散射"; }
    const char* fragmentShaderName() const override { return "advanced_atmosphere.frag.spv"; }

    void configureChapter() override {
        interactive_.camera().setTarget({0.0f, 1.0f, -3.0f});
        interactive_.camera().setDistance(5.0f);
        interactive_.camera().setAngles(0.0f, 4.0f);
    }

    void updateChapter() override {
        if (animateSun_)
            sunAzimuth_ += 0.06f;
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Atmosphere", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Checkbox("Animate sun", &animateSun_);
            ImGui::SliderFloat("Sun elevation", &sunElevation_, -8.0f, 80.0f, "%.1f deg");
            ImGui::SliderFloat("Sun azimuth", &sunAzimuth_, -180.0f, 180.0f, "%.1f deg");
            ImGui::SliderFloat("Sun intensity", &sunIntensity_, 0.2f, 4.0f, "%.2f");
            ImGui::SliderFloat("Rayleigh", &rayleigh_, 0.1f, 2.5f, "%.2f");
            ImGui::SliderFloat("Mie", &mie_, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Mie anisotropy", &anisotropy_, 0.0f, 0.94f, "%.2f");
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_, 0.0f};
        fillCameraFrame(push);
        const float elevation = glm::radians(sunElevation_);
        const float azimuth = glm::radians(sunAzimuth_);
        const glm::vec3 sunDirection{std::cos(elevation) * std::cos(azimuth), std::sin(elevation),
                                     std::cos(elevation) * std::sin(azimuth)};
        push.values[5] = glm::vec4(sunDirection, sunIntensity_);
        push.values[6] = {rayleigh_, mie_, anisotropy_, cameraHeight_};
    }

  private:
    float sunElevation_ = 12.0f;
    float sunAzimuth_ = -36.0f;
    float sunIntensity_ = 1.45f;
    float rayleigh_ = 0.78f;
    float mie_ = 0.32f;
    float anisotropy_ = 0.76f;
    float cameraHeight_ = 2.2f;
    bool animateSun_ = false;
};

int main() {
    try {
        Ch73App app;
        app.run("ch73 - GPU Atmospheric Scattering", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch73 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
