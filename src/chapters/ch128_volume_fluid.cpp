#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>
#include <imgui.h>
#include <iostream>

class Ch128App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第128章：体积与流体"; }
    const char* featureFragmentShader() const override { return "volume_fluid.frag.spv"; }
    const char* featureComputeShader() const override { return "volume_fluid.comp.spv"; }
    uint32_t featureMode() const override { return 128; }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Volume / Fluid",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::TextUnformatted("Froxel fog / volume cloud / FFT ocean spectrum / caustics");
            ImGui::SliderFloat("Density",&density_,0.0f,2.0f);
        } ImGui::End();
    }
    void featureParameters(AdvancedGpuPush& push) const override { push.values[1].x=density_; }
  private: float density_=0.8f;
};
int main(){try{Ch128App app;app.run("ch128 - Volume Fluid",1280,720);}catch(const std::exception& e){std::cerr<<"ch128 failed: "<<e.what()<<'\n';return 1;}return 0;}
