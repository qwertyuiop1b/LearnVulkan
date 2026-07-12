/**
 * @file ch91_physics_integration.cpp
 * @brief Fixed-step GPU rigid bodies rendered from the simulation SSBO.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch91App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第91章：物理/渲染器数据集成"; }
    const char* vertexShaderName() const override { return "engine_physics.vert.spv"; }
    const char* fragmentShaderName() const override { return "engine_physics.frag.spv"; }
    const char* computeShaderName() const override { return "engine_physics.comp.spv"; }
    bool chapterUsesAlphaBlend() const override { return true; }
    uint32_t drawVertexCount() const override { return 6; }
    uint32_t drawInstanceCount() const override { return BODY_COUNT; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        return {BODY_COUNT * sizeof(glm::vec4) * 2, 16, 16, 16};
    }

    glm::uvec3 computeDispatch() const override { return {(BODY_COUNT + 63) / 64, 1, 1}; }

    void configureChapter() override {
        bgColor_ = {0.012f, 0.018f, 0.032f};
        interactive_.camera().setTarget({0.0f, 2.0f, -4.0f});
        interactive_.camera().setDistance(15.0f);
        interactive_.camera().setAngles(0.0f, -0.12f);
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Physics Integration", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderFloat("Gravity", &gravity_, 0.0f, 18.0f);
            ImGui::SliderFloat("Restitution", &restitution_, 0.05f, 0.92f);
            ImGui::Checkbox("Simulate", &simulate_);
            ImGui::Text("%u bodies, fixed 60 Hz compute step", BODY_COUNT);
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        const float aspect = static_cast<float>(extent_.width) / static_cast<float>(std::max(extent_.height, 1u));
        const glm::mat4 vp = interactive_.camera().projectionMatrix(aspect, 55.0f, 0.1f, 100.0f) *
                             interactive_.camera().viewMatrix();
        for (uint32_t column = 0; column < 4; ++column)
            push.values[column] = vp[column];
        push.values[4] = glm::vec4(interactive_.camera().eyePosition(), elapsed_);
        push.values[5] = {gravity_, restitution_, simulate_ ? 1.0f : 0.0f, static_cast<float>(BODY_COUNT)};
    }

  private:
    static constexpr uint32_t BODY_COUNT = 96;
    float gravity_ = 9.8f;
    float restitution_ = 0.68f;
    bool simulate_ = true;
};

int main() {
    try {
        Ch91App app;
        app.run("ch91 - Physics Renderer Integration", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch91 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
