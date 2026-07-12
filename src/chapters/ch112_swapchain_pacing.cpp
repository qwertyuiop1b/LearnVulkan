/**
 * @file ch112_swapchain_pacing.cpp
 * @brief Surface, present pacing and HDR capability inspection; DemoApp uses oldSwapchain on rebuild.
 */

#include <vulkan_tutorial/engine/engineering_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch112App final : public EngineeringGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第112章：交换链与帧节奏"; }
    uint32_t engineeringMode() const override { return 112; }
    glm::vec4 engineeringParameters() const override {
        return {hdrSurface_ ? 1.0f : 0.0f, mailbox_ ? 1.0f : 0.0f,
                profile_.presentWait ? 1.0f : 0.0f, static_cast<float>(surfaceFormatCount_)};
    }

    void onEngineeringInit() override {
        uint32_t count = 0;
        VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physDev_, surface_, &count, nullptr));
        std::vector<VkSurfaceFormatKHR> formats(count);
        VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physDev_, surface_, &count, formats.data()));
        surfaceFormatCount_ = count;
        for (const auto& format : formats) {
            if (format.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT ||
                format.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT)
                hdrSurface_ = true;
        }
        count = 0;
        VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physDev_, surface_, &count, nullptr));
        std::vector<VkPresentModeKHR> modes(count);
        VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physDev_, surface_, &count, modes.data()));
        for (VkPresentModeKHR mode : modes) {
            mailbox_ |= mode == VK_PRESENT_MODE_MAILBOX_KHR;
            immediate_ |= mode == VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Swapchain Pacing", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            drawCapabilitySummary();
            ImGui::Text("oldSwapchain rebuild path: active");
            ImGui::Text("Surface formats: %u  HDR colorspace: %s", surfaceFormatCount_, hdrSurface_ ? "yes" : "no");
            ImGui::Text("FIFO: required  Mailbox: %s  Immediate: %s", mailbox_ ? "yes" : "no",
                        immediate_ ? "yes" : "no");
            ImGui::Text("VK_KHR_present_id: %s", profile_.presentId ? "supported" : "fallback");
            ImGui::Text("VK_KHR_present_wait: %s", profile_.presentWait ? "supported" : "fallback");
            ImGui::TextUnformatted("Resize the window to execute oldSwapchain handoff.");
        }
        ImGui::End();
    }

  private:
    uint32_t surfaceFormatCount_ = 0;
    bool hdrSurface_ = false;
    bool mailbox_ = false;
    bool immediate_ = false;
};

int main() {
    try { Ch112App app; app.run("ch112 - Swapchain Pacing", 1280, 720); }
    catch (const std::exception& error) { std::cerr << "ch112 failed: " << error.what() << '\n'; return 1; }
    return 0;
}
