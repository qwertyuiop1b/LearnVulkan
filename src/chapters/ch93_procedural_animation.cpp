/**
 * @file ch93_procedural_animation.cpp
 * @brief GPU-authored humanoid skeleton with procedural gait and two-bone IK.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch93App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第93章：GPU 程序化骨骼动画"; }
    const char* vertexShaderName() const override { return "engine_animation.vert.spv"; }
    const char* fragmentShaderName() const override { return "engine_animation.frag.spv"; }
    const char* computeShaderName() const override { return "engine_animation.comp.spv"; }
    bool chapterUsesAlphaBlend() const override { return true; }
    uint32_t drawVertexCount() const override { return 6; }
    uint32_t drawInstanceCount() const override { return BONE_COUNT; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        return {BONE_COUNT * sizeof(glm::vec4) * 3, 16, 16, 16};
    }
    glm::uvec3 computeDispatch() const override { return {1, 1, 1}; }

    void configureChapter() override {
        bgColor_ = {0.012f, 0.02f, 0.038f};
        interactive_.camera().setTarget({0.0f, 1.6f, 0.0f});
        interactive_.camera().setDistance(8.0f);
        interactive_.camera().setAngles(0.0f, -0.05f);
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("GPU Animation", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderFloat("Gait speed", &gaitSpeed_, 0.0f, 5.0f);
            ImGui::SliderFloat2("IK target", &ikTarget_.x, -1.5f, 1.5f);
            ImGui::Checkbox("Two-bone IK", &enableIk_);
            ImGui::Text("Compute skeleton -> SSBO -> instanced bone draw");
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        const float aspect = static_cast<float>(extent_.width) / static_cast<float>(std::max(extent_.height, 1u));
        const glm::mat4 vp = interactive_.camera().projectionMatrix(aspect, 52.0f, 0.1f, 60.0f) *
                             interactive_.camera().viewMatrix();
        for (uint32_t column = 0; column < 4; ++column)
            push.values[column] = vp[column];
        push.values[4] = glm::vec4(interactive_.camera().eyePosition(), elapsed_);
        push.values[5] = {gaitSpeed_, ikTarget_.x, ikTarget_.y, enableIk_ ? 1.0f : 0.0f};
    }

  private:
    static constexpr uint32_t BONE_COUNT = 14;
    float gaitSpeed_ = 2.4f;
    glm::vec2 ikTarget_{0.95f, 1.55f};
    bool enableIk_ = true;
};

int main() {
    try {
        Ch93App app;
        app.run("ch93 - GPU Procedural Animation", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch93 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
