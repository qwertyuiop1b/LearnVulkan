/**
 * @file ch114_debug_diagnostics.cpp
 * @brief Debug object names, command-buffer markers and crash-diagnostic capability routing.
 */

#include <vulkan_tutorial/engine/engineering_gpu_chapter.hpp>

#include <cstdlib>
#include <imgui.h>
#include <iostream>

class Ch114App final : public EngineeringGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第114章：调试与崩溃诊断"; }
    uint32_t engineeringMode() const override { return 114; }
    glm::vec4 engineeringParameters() const override {
        return {objectNamed_ ? 1.0f : 0.0f, markerRecorded_ ? 1.0f : 0.0f,
                gpuAssistedRequested_ ? 1.0f : 0.0f, profile_.deviceFault ? 1.0f : 0.0f};
    }

    void onEngineeringInit() override {
        gpuAssistedRequested_ = std::getenv("VK_TUTORIAL_GPU_ASSISTED") != nullptr;
        createBuffer(physDev_, device_, 4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, namedBuffer_, namedMemory_);
        auto setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            vkGetDeviceProcAddr(device_, "vkSetDebugUtilsObjectNameEXT"));
        if (setName) {
            VkDebugUtilsObjectNameInfoEXT name{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
            name.objectType = VK_OBJECT_TYPE_BUFFER;
            name.objectHandle = reinterpret_cast<uint64_t>(namedBuffer_);
            name.pObjectName = "ch114.crash_probe_buffer";
            objectNamed_ = setName(device_, &name) == VK_SUCCESS;
        }

        auto beginLabel = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            vkGetDeviceProcAddr(device_, "vkCmdBeginDebugUtilsLabelEXT"));
        auto endLabel = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            vkGetDeviceProcAddr(device_, "vkCmdEndDebugUtilsLabelEXT"));
        if (beginLabel && endLabel) {
            VkCommandBuffer command = beginSingleTimeCommands(device_, cmdPool_);
            VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
            label.pLabelName = "ch114 diagnostics marker";
            label.color[0] = 1.0f; label.color[1] = 0.2f; label.color[2] = 0.05f; label.color[3] = 1.0f;
            beginLabel(command, &label);
            vkCmdFillBuffer(command, namedBuffer_, 0, VK_WHOLE_SIZE, 0xD1A6C0DEu);
            endLabel(command);
            endSingleTimeCommands(device_, cmdPool_, gQueue_, command);
            markerRecorded_ = true;
        }
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("GPU Diagnostics", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            drawCapabilitySummary();
            status("Debug object name", objectNamed_);
            status("GPU command marker", markerRecorded_);
            status("GPU-assisted validation requested", gpuAssistedRequested_);
            status("VK_EXT_device_fault", profile_.deviceFault);
            ImGui::TextUnformatted("Set VK_TUTORIAL_GPU_ASSISTED=1 before launch to request validation mode.");
            ImGui::TextUnformatted("Device-fault collection is enabled only on drivers exposing the extension.");
        }
        ImGui::End();
    }

    void onEngineeringShutdown() override {
        vkDestroyBuffer(device_, namedBuffer_, nullptr);
        vkFreeMemory(device_, namedMemory_, nullptr);
    }

  private:
    VkBuffer namedBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory namedMemory_ = VK_NULL_HANDLE;
    bool objectNamed_ = false;
    bool markerRecorded_ = false;
    bool gpuAssistedRequested_ = false;

    static void status(const char* label, bool active) {
        ImGui::TextColored(active ? ImVec4(0.2f, 1.0f, 0.45f, 1.0f) : ImVec4(1.0f, 0.55f, 0.2f, 1.0f),
                           "%s: %s", label, active ? "active" : "fallback/unavailable");
    }
};

int main() {
    try { Ch114App app; app.run("ch114 - Debug Diagnostics", 1280, 720); }
    catch (const std::exception& error) { std::cerr << "ch114 failed: " << error.what() << '\n'; return 1; }
    return 0;
}
