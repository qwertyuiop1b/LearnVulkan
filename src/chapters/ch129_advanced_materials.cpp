#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>
#include <imgui.h>
#include <iostream>

class Ch129App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第129章：高级材质"; }
    const char* featureFragmentShader() const override { return "advanced_material.frag.spv"; }
    const char* featureComputeShader() const override { return "advanced_material.comp.spv"; }
    uint32_t featureMode() const override { return 129; }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Advanced Materials",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::TextUnformatted("Clearcoat / Transmission / Sheen / Anisotropy / Iridescence");
            ImGui::SliderFloat("Transmission",&transmission_,0.0f,1.0f);
        } ImGui::End();
    }
    void featureParameters(AdvancedGpuPush& push) const override { push.values[1].x=transmission_; }
  private: float transmission_=0.55f;
};
int main(){try{Ch129App app;app.run("ch129 - Advanced Materials",1280,720);}catch(const std::exception& e){std::cerr<<"ch129 failed: "<<e.what()<<'\n';return 1;}return 0;}
