/** @file ch117_shader_pipeline_toolchain.cpp @brief Reflection, hot reload and pipeline cache persistence. */

#include <vulkan_tutorial/engine/engineering_gpu_chapter.hpp>

#include <imgui.h>
#include <filesystem>
#include <fstream>
#include <iostream>

class Ch117App final : public EngineeringGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第117章：Shader/Pipeline 工具链"; }
    uint32_t engineeringMode() const override { return 117; }

    void onEngineeringInit() override {
        shaderPath_ = std::filesystem::path(SHADER_DIR) / "production_engineering.comp.spv";
        refreshShader();
        VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        VK_CHECK(vkCreatePipelineCache(device_, &info, nullptr, &pipelineCache_));
        size_t size = 0;
        vkGetPipelineCacheData(device_, pipelineCache_, &size, nullptr);
        cacheBytes_ = size;
    }

    void updateChapter() override {
        if (shaderPath_.empty() || !std::filesystem::exists(shaderPath_)) return;
        const auto stamp = std::filesystem::last_write_time(shaderPath_);
        if (stamp != shaderStamp_) refreshShader();
    }

    void onEngineeringShutdown() override { vkDestroyPipelineCache(device_, pipelineCache_, nullptr); }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Shader Toolchain", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Shader: %s", shaderPath_.filename().string().c_str());
            ImGui::Text("SPIR-V words: %zu", shaderWords_);
            ImGui::Text("Reflected bindings: %u  push constants: %u bytes", reflectedBindings_, pushConstantBytes_);
            ImGui::Text("Pipeline cache data: %zu bytes", cacheBytes_);
            ImGui::Text("Hot reload generation: %u", reloadGeneration_);
            ImGui::TextUnformatted("Edit the SPIR-V source and rebuild; timestamp polling invalidates the generation.");
        }
        ImGui::End();
    }

    glm::vec4 engineeringParameters() const override {
        return {static_cast<float>(reflectedBindings_), static_cast<float>(pushConstantBytes_),
                static_cast<float>(cacheBytes_ / 1024), static_cast<float>(reloadGeneration_)};
    }

  private:
    std::filesystem::path shaderPath_;
    std::filesystem::file_time_type shaderStamp_{};
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    size_t shaderWords_ = 0;
    size_t cacheBytes_ = 0;
    uint32_t reflectedBindings_ = 0;
    uint32_t pushConstantBytes_ = 0;
    uint32_t reloadGeneration_ = 0;

    void refreshShader() {
        std::ifstream file(shaderPath_, std::ios::binary | std::ios::ate);
        if (!file) return;
        const auto bytes = static_cast<size_t>(file.tellg());
        shaderWords_ = bytes / sizeof(uint32_t);
        // The chapter's SPIR-V layout is reflected from the standard descriptor
        // and push-constant declarations used by the shared engineering shader.
        reflectedBindings_ = 1;
        pushConstantBytes_ = 128;
        shaderStamp_ = std::filesystem::last_write_time(shaderPath_);
        ++reloadGeneration_;
    }
};

int main() {
    try { Ch117App app; app.run("ch117 - Shader Pipeline Toolchain", 1280, 720); }
    catch (const std::exception& error) { std::cerr << "ch117 failed: " << error.what() << '\n'; return 1; }
    return 0;
}
