#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>
#include <imgui.h>
#include <iostream>

class Ch121App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第121章：生产级 GPU Driven"; }
    const char* featureFragmentShader() const override { return "gpu_driven.frag.spv"; }
    const char* featureComputeShader() const override { return "gpu_driven_lod.comp.spv"; }
    const char* vertexShaderName() const override { return "gpu_driven.vert.spv"; }
    uint32_t featureMode() const override { return 121; }
    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> featureStorageBufferSizes() const override {
        return {1024 * sizeof(glm::vec4), 1024 * sizeof(VkDrawIndirectCommand), sizeof(uint32_t), 16};
    }
    std::array<VkBufferUsageFlags, STORAGE_BUFFER_COUNT> featureExtraBufferUsage() const override {
        return {0, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, 0};
    }
    int32_t indirectDrawBufferIndex() const override { return 1; }
    int32_t indirectDrawCountBufferIndex() const override { return 2; }
    uint32_t indirectMaxDrawCount() const override { return 1024; }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("GPU Driven",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::TextUnformatted("Visibility / LOD / material bucket streams are generated on GPU.");
            ImGui::Text("Objects: 1024  Buckets: 8  Hi-Z fallback: analytic");
        } ImGui::End();
    }
};
int main(){try{Ch121App app;app.run("ch121 - Production GPU Driven",1280,720);}catch(const std::exception& e){std::cerr<<"ch121 failed: "<<e.what()<<'\n';return 1;}return 0;}
