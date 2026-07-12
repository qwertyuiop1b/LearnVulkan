/**
 * @file ch77_virtual_texture.cpp
 * @brief GPU page-table residency and virtual-texture sampling visualization.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <cmath>
#include <iostream>

class Ch77App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第77章：GPU 虚拟纹理"; }
    const char* fragmentShaderName() const override { return "advanced_virtual_texture.frag.spv"; }
    const char* computeShaderName() const override { return "advanced_virtual_texture.comp.spv"; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        return {64u * 64u * sizeof(uint32_t), 16, 16, 16};
    }

    glm::uvec3 computeDispatch() const override { return {8, 8, 1}; }

    void updateChapter() override {
        if (animateFocus_) {
            focus_.x = 0.5f + std::sin(elapsed_ * 0.31f) * 0.23f;
            focus_.y = 0.5f + std::cos(elapsed_ * 0.27f) * 0.21f;
        }
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Virtual Texture", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Checkbox("Animate focus", &animateFocus_);
            ImGui::SliderFloat2("Feedback focus", &focus_.x, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Resident budget", &budget_, 0.05f, 1.0f, "%.2f");
            ImGui::SliderFloat("View zoom", &zoom_, 0.15f, 1.0f, "%.2f");
        }
        ImGui::End();
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_, zoom_};
        push.values[1] = {focus_.x, focus_.y, budget_, 0.0f};
    }

  private:
    glm::vec2 focus_{0.5f, 0.5f};
    float budget_ = 0.42f;
    float zoom_ = 0.58f;
    bool animateFocus_ = true;
};

int main() {
    try {
        Ch77App app;
        app.run("ch77 - GPU Virtual Texture", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch77 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
