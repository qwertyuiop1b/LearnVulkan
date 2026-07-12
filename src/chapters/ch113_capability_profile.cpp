/**
 * @file ch113_capability_profile.cpp
 * @brief Features2-derived capability profile and explicit platform downgrade policy.
 */

#include <vulkan_tutorial/engine/engineering_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch113App final : public EngineeringGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第113章：Vulkan 1.4 与能力 Profile"; }
    uint32_t engineeringMode() const override { return 113; }
    glm::vec4 engineeringParameters() const override {
        return {profile_.synchronization2 ? 1.0f : 0.0f, profile_.descriptorIndexing ? 1.0f : 0.0f,
                profile_.rayQuery ? 1.0f : 0.0f, profile_.meshShader ? 1.0f : 0.0f};
    }

    void onEngineeringInit() override {
#ifdef __APPLE__
        platform_ = "MoltenVK";
#elif defined(__ANDROID__)
        platform_ = "Android";
#else
        platform_ = "Desktop Vulkan";
#endif
        if (profile_.rayQuery && profile_.accelerationStructure)
            rendererPath_ = "Hybrid RT";
        else if (profile_.descriptorIndexing && profile_.bufferDeviceAddress)
            rendererPath_ = "GPU Driven Raster";
        else
            rendererPath_ = "Core Raster Fallback";
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 280}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Capability Profile", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            drawCapabilitySummary();
            ImGui::Text("Platform: %s", platform_);
            ImGui::Text("Selected renderer: %s", rendererPath_);
            ImGui::Separator();
            capability("Timeline / Sync2", profile_.timelineSemaphore && profile_.synchronization2);
            capability("Dynamic Rendering", profile_.dynamicRendering);
            capability("Descriptor Indexing", profile_.descriptorIndexing);
            capability("Buffer Device Address", profile_.bufferDeviceAddress);
            capability("Mesh Shader", profile_.meshShader);
            capability("Ray Query + AS", profile_.rayQuery && profile_.accelerationStructure);
            capability("Memory Budget", profile_.memoryBudget);
            ImGui::Text("Vulkan 1.4 native: %s",
                        profile_.apiVersion >= VK_MAKE_API_VERSION(0, 1, 4, 0) ? "yes" : "1.3 compatibility profile");
        }
        ImGui::End();
    }

  private:
    const char* platform_ = "Unknown";
    const char* rendererPath_ = "Core Raster Fallback";

    static void capability(const char* label, bool supported) {
        ImGui::TextColored(supported ? ImVec4(0.25f, 1.0f, 0.5f, 1.0f) : ImVec4(1.0f, 0.55f, 0.2f, 1.0f),
                           "%s: %s", label, supported ? "native" : "fallback");
    }
};

int main() {
    try { Ch113App app; app.run("ch113 - Capability Profile", 1280, 720); }
    catch (const std::exception& error) { std::cerr << "ch113 failed: " << error.what() << '\n'; return 1; }
    return 0;
}
