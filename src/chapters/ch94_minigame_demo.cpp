/**
 * @file ch94_minigame_demo.cpp
 * @brief Compact real GPU capstone: simulation state feeds a rendered world.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch94App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第94章：GPU MiniGame 综合场景"; }
    const char* fragmentShaderName() const override { return "engine_minigame.frag.spv"; }
    const char* computeShaderName() const override { return "engine_minigame.comp.spv"; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        return {ENTITY_COUNT * sizeof(glm::vec4) * 2, 16, 16, 16};
    }
    glm::uvec3 computeDispatch() const override { return {(ENTITY_COUNT + 63) / 64, 1, 1}; }

    void configureChapter() override {
        bgColor_ = {0.008f, 0.014f, 0.025f};
        interactive_.camera().setTarget({0.0f, 0.8f, -4.0f});
        interactive_.camera().setDistance(11.0f);
        interactive_.camera().setAngles(0.0f, -0.2f);
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("MiniGame Systems", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderInt("Active enemies", &enemyCount_, 4, static_cast<int>(ENTITY_COUNT - 1));
            ImGui::SliderFloat("World speed", &worldSpeed_, 0.0f, 2.0f);
            ImGui::SliderFloat("Weather", &weather_, 0.0f, 1.0f);
            ImGui::TextUnformatted("Compute entities + terrain + lighting + weather particles");
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_,
                          static_cast<float>(enemyCount_)};
        push.values[1] = {worldSpeed_, weather_, static_cast<float>(ENTITY_COUNT), 0.0f};
        fillCameraFrame(push, 2, 55.0f);
    }

  private:
    static constexpr uint32_t ENTITY_COUNT = 48;
    int enemyCount_ = 28;
    float worldSpeed_ = 0.75f;
    float weather_ = 0.35f;
};

int main() {
    try {
        Ch94App app;
        app.run("ch94 - GPU MiniGame Capstone", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch94 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
