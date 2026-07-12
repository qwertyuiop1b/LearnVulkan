#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>
#include <vulkan_tutorial/meshlet_builder.hpp>
#include <imgui.h>
#include <iostream>

class Ch122App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第122章：Meshlet"; }
    const char* featureFragmentShader() const override { return "meshlet_cull.frag.spv"; }
    const char* featureComputeShader() const override { return "meshlet_cull.comp.spv"; }
    uint32_t featureMode() const override { return 122; }
    void onAdvancedInit() override {
        std::vector<glm::vec3> vertices;
        std::vector<uint32_t> indices;
        for (uint32_t y = 0; y <= 32; ++y)
            for (uint32_t x = 0; x <= 32; ++x)
                vertices.emplace_back(float(x) / 16.0f - 1.0f, 0.0f, float(y) / 16.0f - 1.0f);
        for (uint32_t y = 0; y < 32; ++y)
            for (uint32_t x = 0; x < 32; ++x) {
                uint32_t i = y * 33 + x;
                indices.insert(indices.end(), {i, i + 1, i + 33, i + 1, i + 34, i + 33});
            }
        meshlets_ = vulkan_tutorial::buildOfflineMeshlets(vertices, indices);
    }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Meshlet Culling",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::TextUnformatted("Cone + frustum + Hi-Z visibility masks");
            ImGui::Text("Offline clusters: %zu (64 triangles/cluster)", meshlets_.size());
            ImGui::TextUnformatted("Cone + frustum + Hi-Z tests feed the raster fallback.");
        } ImGui::End();
    }
  private:
    std::vector<vulkan_tutorial::OfflineMeshlet> meshlets_;
};
int main(){try{Ch122App app;app.run("ch122 - Meshlet Culling",1280,720);}catch(const std::exception& e){std::cerr<<"ch122 failed: "<<e.what()<<'\n';return 1;}return 0;}
