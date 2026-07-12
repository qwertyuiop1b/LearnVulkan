/**
 * @file ch75_lut_color_grading.cpp
 * @brief Compute-generated 3D LUT with trilinear shader sampling.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch75App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第75章：GPU 3D LUT 调色"; }
    const char* fragmentShaderName() const override { return "advanced_lut.frag.spv"; }
    const char* computeShaderName() const override { return "advanced_lut.comp.spv"; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        return {VkDeviceSize(LUT_SIZE) * LUT_SIZE * LUT_SIZE * sizeof(glm::vec4), 16, 16, 16};
    }

    glm::uvec3 computeDispatch() const override { return {LUT_SIZE / 4, LUT_SIZE / 4, LUT_SIZE / 4}; }

    void buildChapterUi() override {
        static const char* presets[] = {"Neutral", "Cinematic", "Night vision", "Warm", "Cool"};
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("3D LUT", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Combo("Preset", &preset_, presets, 5);
            ImGui::SliderFloat("Blend", &blend_, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Contrast", &contrast_, 0.55f, 1.65f, "%.2f");
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(LUT_SIZE), static_cast<float>(preset_), blend_, contrast_};
        push.values[1] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_, 0.0f};
    }

  private:
    static constexpr uint32_t LUT_SIZE = 32;
    int preset_ = 1;
    float blend_ = 0.88f;
    float contrast_ = 1.08f;
};

int main() {
    try {
        Ch75App app;
        app.run("ch75 - GPU 3D LUT Color Grading", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch75 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
