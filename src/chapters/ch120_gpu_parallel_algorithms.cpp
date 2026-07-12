#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>
#include <imgui.h>
#include <iostream>

class Ch120App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第120章：GPU 并行算法"; }
    const char* featureFragmentShader() const override { return "parallel_algorithms.frag.spv"; }
    const char* featureComputeShader() const override { return "parallel_algorithms.comp.spv"; }
    uint32_t featureMode() const override { return 120; }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Parallel Algorithms",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::TextUnformatted("Reduction / Prefix Scan / Stream Compaction / Radix key lanes");
            ImGui::Text("Work items: 1024  Workgroup: 64");
        } ImGui::End();
    }
};
int main(){try{Ch120App app;app.run("ch120 - GPU Parallel Algorithms",1280,720);}catch(const std::exception& e){std::cerr<<"ch120 failed: "<<e.what()<<'\n';return 1;}return 0;}
