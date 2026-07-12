#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>
#include <imgui.h>
#include <iostream>

class Ch126App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第126章：光追降噪与 GI"; }
    const char* featureFragmentShader() const override { return "denoise_gi.frag.spv"; }
    const char* featureComputeShader() const override { return "denoise_gi.comp.spv"; }
    uint32_t featureMode() const override { return 126; }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("RT Denoise / GI",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::TextUnformatted("SVGF temporal variance -> A-Trous -> DDGI / ReSTIR GI");
            ImGui::SliderInt("A-Trous iterations",&iterations_,1,5);
        } ImGui::End();
    }
    void featureParameters(AdvancedGpuPush& push) const override { push.values[1].x=static_cast<float>(iterations_); }
  private: int iterations_=3;
};
int main(){try{Ch126App app;app.run("ch126 - Ray Denoise GI",1280,720);}catch(const std::exception& e){std::cerr<<"ch126 failed: "<<e.what()<<'\n';return 1;}return 0;}
