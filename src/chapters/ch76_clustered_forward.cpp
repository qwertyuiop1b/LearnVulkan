/**
 * @file ch76_clustered_forward.cpp
 * @brief Compute-built clustered light lists consumed by the fragment shader.
 */

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch76App final : public AdvancedGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第76章：GPU Clustered Forward+"; }
    const char* fragmentShaderName() const override { return "advanced_cluster.frag.spv"; }
    const char* computeShaderName() const override { return "advanced_cluster.comp.spv"; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const override {
        const VkDeviceSize clusterCount = CLUSTERS_X * CLUSTERS_Y * CLUSTERS_Z;
        return {clusterCount * sizeof(uint32_t),
                clusterCount * MAX_LIGHTS_PER_CLUSTER * sizeof(uint32_t), 16, 16};
    }

    glm::uvec3 computeDispatch() const override { return {CLUSTERS_X, CLUSTERS_Y, CLUSTERS_Z}; }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Clustered Forward+", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderInt("Point lights", &lightCount_, 8, 128);
            ImGui::Checkbox("Animate lights", &animate_);
        }
        ImGui::End();
    }

    void updateChapter() override {
        if (!animate_)
            elapsed_ -= 0.016f;
    }

    void fillPushConstants(AdvancedGpuPush& push) const override {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_,
                          static_cast<float>(lightCount_)};
        push.values[1] = {static_cast<float>(CLUSTERS_X), static_cast<float>(CLUSTERS_Y),
                          static_cast<float>(CLUSTERS_Z), static_cast<float>(MAX_LIGHTS_PER_CLUSTER)};
    }

  private:
    static constexpr uint32_t CLUSTERS_X = 16;
    static constexpr uint32_t CLUSTERS_Y = 9;
    static constexpr uint32_t CLUSTERS_Z = 24;
    static constexpr uint32_t MAX_LIGHTS_PER_CLUSTER = 24;
    int lightCount_ = 96;
    bool animate_ = true;
};

int main() {
    try {
        Ch76App app;
        app.run("ch76 - GPU Clustered Forward+", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch76 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
