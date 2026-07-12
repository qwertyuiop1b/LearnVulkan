#include <vulkan_tutorial/engine/feature_gpu_chapter.hpp>

#include <imgui.h>
#include <filesystem>
#include <iostream>

class Ch130App final : public FeatureGpuChapter {
  protected:
    const char* chapterTitle() const override { return "第130章：资产管线"; }
    const char* featureFragmentShader() const override { return "advanced_material.frag.spv"; }
    const char* featureComputeShader() const override { return "advanced_material.comp.spv"; }
    uint32_t featureMode() const override { return 130; }
    void onAdvancedInit() override { scanAssets(); queryFormats(); }
    void featureUi() override {
        ImGui::SetNextWindowPos({12,315},ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Asset Pipeline",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::Text("Files scanned: %u",files_);
            ImGui::Text("KTX2/Basis: %u  glTF: %u  compressed candidates: %u",ktx_,gltf_,compressed_);
            ImGui::Text("BC7: %s  ASTC: %s  ETC2: %s", bc7_ ? "supported" : "fallback",
                        astc_ ? "supported" : "fallback", etc2_ ? "supported" : "fallback");
            ImGui::TextUnformatted("Runtime path: KTX2/BasisU -> BC/ASTC/ETC2 profile -> meshopt/glTF upload.");
        } ImGui::End();
    }
  private:
    uint32_t files_=0,ktx_=0,gltf_=0,compressed_=0;
    bool bc7_=false, astc_=false, etc2_=false;
    void scanAssets(){
        const std::filesystem::path root=ASSET_DIR;
        if(!std::filesystem::exists(root))return;
        for(const auto& entry:std::filesystem::recursive_directory_iterator(root)){
            if(!entry.is_regular_file())continue; ++files_;
            const std::string ext=entry.path().extension().string();
            if(ext==".ktx2"||ext==".basis")++ktx_;
            if(ext==".gltf"||ext==".glb")++gltf_;
            if(ext==".dds"||ext==".ktx2"||ext==".astc")++compressed_;
        }
    }

    void queryFormats() {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physDev_, VK_FORMAT_BC7_UNORM_BLOCK, &properties);
        bc7_ = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
        vkGetPhysicalDeviceFormatProperties(physDev_, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, &properties);
        astc_ = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
        vkGetPhysicalDeviceFormatProperties(physDev_, VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, &properties);
        etc2_ = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
    }
};
int main(){try{Ch130App app;app.run("ch130 - Asset Pipeline",1280,720);}catch(const std::exception& e){std::cerr<<"ch130 failed: "<<e.what()<<'\n';return 1;}return 0;}
