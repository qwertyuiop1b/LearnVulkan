#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>
#include <imgui.h>
#include <iostream>

class Ch119App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第119章：Reverse-Z 与 Hi-Z"; }
    const char* featureFragmentShader() const override { return "reverse_z_hiz.frag.spv"; }
    const char* featureComputeShader() const override { return "reverse_z_hiz.comp.spv"; }
    uint32_t featureMode() const override { return 119; }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Reverse-Z / Hi-Z",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::Checkbox("Infinite far plane",&infiniteFar_);
            ImGui::Checkbox("Show depth pyramid",&showPyramid_);
            ImGui::Text("Depth convention: %s",infiniteFar_?"GREATER, far=0":"GREATER");
        } ImGui::End();
    }
    void featureParameters(AdvancedGpuPush& push) const override { push.values[1]={infiniteFar_?1.f:0.f,showPyramid_?1.f:0.f,0,0}; }
  private: bool infiniteFar_=true,showPyramid_=true;
};
int main(){try{Ch119App app;app.run("ch119 - Reverse Z Hi-Z",1280,720);}catch(const std::exception& e){std::cerr<<"ch119 failed: "<<e.what()<<'\n';return 1;}return 0;}
