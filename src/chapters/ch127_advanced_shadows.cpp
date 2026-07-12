#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>
#include <imgui.h>
#include <iostream>

class Ch127App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第127章：高级阴影"; }
    const char* featureFragmentShader() const override { return "advanced_shadow.frag.spv"; }
    const char* featureComputeShader() const override { return "advanced_shadow.comp.spv"; }
    uint32_t featureMode() const override { return 127; }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Advanced Shadows",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::TextUnformatted("EVSM / MSM moments / point-light atlas / virtual shadow pages");
            ImGui::SliderFloat("Bleed reduction",&bleed_,0.0f,1.0f);
        } ImGui::End();
    }
    void featureParameters(AdvancedGpuPush& push) const override { push.values[1].x=bleed_; }
  private: float bleed_=0.2f;
};
int main(){try{Ch127App app;app.run("ch127 - Advanced Shadows",1280,720);}catch(const std::exception& e){std::cerr<<"ch127 failed: "<<e.what()<<'\n';return 1;}return 0;}
