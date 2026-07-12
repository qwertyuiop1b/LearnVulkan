#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>
#include <imgui.h>
#include <iostream>

class Ch124App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第124章：TAAU 与动态分辨率"; }
    const char* featureFragmentShader() const override { return "taau_reconstruct.frag.spv"; }
    const char* featureComputeShader() const override { return "taau_reconstruct.comp.spv"; }
    uint32_t featureMode() const override { return 124; }
    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> featureStorageBufferSizes() const override {
        return {1024 * sizeof(glm::vec4), 1024 * sizeof(glm::vec4), 16, 16};
    }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("TAAU / Dynamic Resolution",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::SliderFloat("Render scale",&renderScale_,0.5f,1.0f);
            ImGui::SliderFloat("Sharpen",&sharpen_,0.0f,1.0f);
            ImGui::Text("History reconstruction and reactive sharpen run in Compute.");
        } ImGui::End();
    }
    void featureParameters(AdvancedGpuPush& push) const override { push.values[1]={renderScale_,sharpen_,0,0}; }
  private: float renderScale_=0.78f,sharpen_=0.35f;
};
int main(){try{Ch124App app;app.run("ch124 - TAAU Dynamic Resolution",1280,720);}catch(const std::exception& e){std::cerr<<"ch124 failed: "<<e.what()<<'\n';return 1;}return 0;}
