/**
 * @file ch83_restir.cpp
 * @brief GPU temporal and spatial reservoir resampling for many-light shading.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch83App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第83章：GPU ReSTIR DI"; }
    const char* fragmentShaderName() const override { return "advanced_restir.frag.spv"; }
    const char* computeShaderName() const override { return "advanced_restir.comp.spv"; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        const VkDeviceSize reservoirSize = VkDeviceSize(RESERVOIR_WIDTH) * RESERVOIR_HEIGHT * sizeof(glm::vec4) * 2;
        return {reservoirSize, reservoirSize, 16, 16};
    }

    glm::uvec3 computeDispatch() const override {
        return {(RESERVOIR_WIDTH + 7) / 8, (RESERVOIR_HEIGHT + 7) / 8, 1};
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("ReSTIR DI", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderInt("Analytic lights", &lightCount_, 8, 512);
            ImGui::SliderInt("Spatial radius", &spatialRadius_, 1, 12);
            ImGui::SliderFloat("Temporal reuse", &temporalReuse_, 0.0f, 0.98f, "%.2f");
            ImGui::SliderFloat("Spatial reuse", &spatialReuse_, 0.0f, 2.0f, "%.2f");
            ImGui::Checkbox("Reservoir debug", &debugReservoirs_);
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_,
                          static_cast<float>(frameNumber_ & 1u)};
        push.values[1] = {static_cast<float>(RESERVOIR_WIDTH), static_cast<float>(RESERVOIR_HEIGHT),
                          static_cast<float>(lightCount_), static_cast<float>(spatialRadius_)};
        push.values[2] = {temporalReuse_, spatialReuse_, debugReservoirs_ ? 1.0f : 0.0f, 173.0f};
    }

  private:
    static constexpr uint32_t RESERVOIR_WIDTH = 160;
    static constexpr uint32_t RESERVOIR_HEIGHT = 90;
    int lightCount_ = 256;
    int spatialRadius_ = 5;
    float temporalReuse_ = 0.90f;
    float spatialReuse_ = 0.72f;
    bool debugReservoirs_ = false;
};

int main() {
    try {
        Ch83App app;
        app.run("ch83 - GPU ReSTIR DI", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch83 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
