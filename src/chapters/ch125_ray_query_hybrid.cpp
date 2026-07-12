#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>
#include <vulkan_tutorial/engine/production_runtime.hpp>
#include <imgui.h>
#include <iostream>

class Ch125App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第125章：Ray Query 混合渲染"; }
    const char* featureFragmentShader() const override { return "ray_query_hybrid.frag.spv"; }
    const char* featureComputeShader() const override { return "ray_query_hybrid.comp.spv"; }
    uint32_t featureMode() const override { return 125; }
    void onAdvancedInit() override { profile_=vulkan_tutorial::production::CapabilityProfile::query(physDev_); }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Hybrid Ray Query",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::Text("Ray Query + AS: %s",profile_.rayQuery&&profile_.accelerationStructure?"native":"analytic fallback");
            ImGui::TextUnformatted("RT shadow / RTAO / reflection lanes are combined in the lighting buffer.");
        } ImGui::End();
    }
  private: vulkan_tutorial::production::CapabilityProfile profile_{};
};
int main(){try{Ch125App app;app.run("ch125 - Ray Query Hybrid",1280,720);}catch(const std::exception& e){std::cerr<<"ch125 failed: "<<e.what()<<'\n';return 1;}return 0;}
