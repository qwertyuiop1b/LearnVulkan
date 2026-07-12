#pragma once

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>
#include <vulkan_tutorial/engine/production_runtime.hpp>

#include <imgui.h>

/**
 * Shared GPU-backed status surface for the production-engineering chapters.
 * Each chapter owns its Vulkan operation; this class only supplies the
 * visible compute/fragment graph and capability discovery.
 */
class EngineeringGpuChapterApp : public AdvancedGpuChapterApp {
  protected:
    const char* fragmentShaderName() const final { return "production_engineering.frag.spv"; }
    const char* computeShaderName() const final { return "production_engineering.comp.spv"; }

    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const final {
        return {256 * sizeof(glm::vec4), 16, 16, 16};
    }

    glm::uvec3 computeDispatch() const final { return {4, 1, 1}; }

    virtual uint32_t engineeringMode() const = 0;
    virtual glm::vec4 engineeringParameters() const { return {}; }
    virtual void onEngineeringInit() {}
    virtual void onEngineeringShutdown() {}

    void onAdvancedInit() final {
        profile_ = vulkan_tutorial::production::CapabilityProfile::query(physDev_);
        onEngineeringInit();
    }

    void onAdvancedShutdown() final { onEngineeringShutdown(); }

    void fillPushConstants(AdvancedGpuPush& push) const final {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_,
                          static_cast<float>(engineeringMode())};
        push.values[1] = engineeringParameters();
    }

    void configureChapter() override { bgColor_ = {0.008f, 0.012f, 0.022f}; }

    void drawCapabilitySummary() const {
        ImGui::Text("GPU: %s", profile_.deviceName.c_str());
        ImGui::Text("Vulkan %u.%u.%u", VK_API_VERSION_MAJOR(profile_.apiVersion),
                    VK_API_VERSION_MINOR(profile_.apiVersion), VK_API_VERSION_PATCH(profile_.apiVersion));
        ImGui::Separator();
    }

    vulkan_tutorial::production::CapabilityProfile profile_{};
};

using EngineeringGpuChapter = EngineeringGpuChapterApp;
