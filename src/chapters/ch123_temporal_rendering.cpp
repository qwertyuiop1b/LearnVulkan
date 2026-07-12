#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>
#include <imgui.h>
#include <iostream>

class Ch123App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第123章：时域渲染基础"; }
    const char* featureFragmentShader() const override { return "temporal_reprojection.frag.spv"; }
    const char* featureComputeShader() const override { return "temporal_reprojection.comp.spv"; }
    uint32_t featureMode() const override { return 123; }
    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> featureStorageBufferSizes() const override {
        return {1024 * sizeof(glm::vec4), 1024 * sizeof(glm::vec4), 16, 16};
    }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Temporal Rendering",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::TextUnformatted("Jitter / Motion Vector / Reprojection / Disocclusion / Reactive Mask");
            ImGui::SliderFloat("History weight",&historyWeight_,0.0f,1.0f);
        } ImGui::End();
    }
    void featureParameters(AdvancedGpuPush& push) const override { push.values[1]={historyWeight_,0,0,0}; }
  private: float historyWeight_=0.92f;
};
int main(){try{Ch123App app;app.run("ch123 - Temporal Rendering",1280,720);}catch(const std::exception& e){std::cerr<<"ch123 failed: "<<e.what()<<'\n';return 1;}return 0;}
