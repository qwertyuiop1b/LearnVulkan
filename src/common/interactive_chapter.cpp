#include <vulkan_tutorial/interactive_chapter.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <cstring>
#include <fstream>
#include <string>

namespace vulkan_tutorial {

namespace {

InteractiveChapterTools* gTools = nullptr;

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (gTools && gTools->hasUi())
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (gTools && (!gTools->hasUi() || !ImGui::GetIO().WantCaptureMouse)) {
        double x = 0.0;
        double y = 0.0;
        glfwGetCursorPos(window, &x, &y);
        gTools->camera().onMouseButton(button, action, x, y);
    }
}

void cursorPosCallback(GLFWwindow* window, double x, double y)
{
    if (gTools && gTools->hasUi())
        ImGui_ImplGlfw_CursorPosCallback(window, x, y);
    if (gTools && (!gTools->hasUi() || !ImGui::GetIO().WantCaptureMouse))
        gTools->camera().onCursorMove(x, y);
}

void scrollCallback(GLFWwindow* window, double xOffset, double yOffset)
{
    if (gTools && gTools->hasUi())
        ImGui_ImplGlfw_ScrollCallback(window, xOffset, yOffset);
    if (gTools && (!gTools->hasUi() || !ImGui::GetIO().WantCaptureMouse))
        gTools->camera().onScroll(yOffset);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (gTools && gTools->hasUi())
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (gTools && (!gTools->hasUi() || !ImGui::GetIO().WantCaptureKeyboard))
        gTools->camera().onKey(key, action);
}

void charCallback(GLFWwindow* window, unsigned int c)
{
    if (gTools && gTools->hasUi())
        ImGui_ImplGlfw_CharCallback(window, c);
}

} // namespace

void InteractiveChapterTools::attachInput(GLFWwindow* window)
{
    info_.window = window;
    gTools = this;
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCharCallback(window, charCallback);
    const char* title = glfwGetWindowTitle(window);
    if (title)
        windowTitleBase_ = title;
}

bool InteractiveChapterTools::initVulkan(const InteractiveInitInfo& info)
{
    info_ = info;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    // ── 加载中文字体（支持汉字显示）──────────────────────────────────────────
    // 按优先级尝试多个字体路径（macOS / Linux）
    static const char* kCjkFontCandidates[] = {
        // macOS
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        // Linux (Noto CJK)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        nullptr
    };

    bool cjkLoaded = false;
    for (int k = 0; kCjkFontCandidates[k] != nullptr; ++k) {
        if (!std::ifstream(kCjkFontCandidates[k]).good())
            continue;
        // 第 0 步：先加载 ASCII 字形（保证英文清晰）
        ImFontConfig latinCfg;
        latinCfg.SizePixels = 16.0f;
        latinCfg.OversampleH = 2;
        latinCfg.OversampleV = 2;
        io.Fonts->AddFontFromFileTTF(
            kCjkFontCandidates[k], 16.0f, &latinCfg,
            io.Fonts->GetGlyphRangesDefault());
        // 第 1 步：Merge 中文字形到同一字体（合并模式）
        ImFontConfig cjkCfg;
        cjkCfg.MergeMode    = true;
        cjkCfg.OversampleH  = 1;   // CJK 不需要高倍过采样
        cjkCfg.OversampleV  = 1;
        cjkCfg.GlyphOffset  = {0.0f, 1.0f};  // 微调垂直对齐
        io.Fonts->AddFontFromFileTTF(
            kCjkFontCandidates[k], 16.0f, &cjkCfg,
            io.Fonts->GetGlyphRangesChineseFull());
        io.Fonts->Build();
        cjkLoaded = true;
        break;
    }
    if (!cjkLoaded) {
        // 未找到中文字体，使用内置默认字体（中文显示为方块）
        io.Fonts->AddFontDefault();
    }

    ImGui_ImplGlfw_InitForVulkan(info_.window, false);
    gpuProfiler_.init(info_.physicalDevice, info_.device, info_.maxFramesInFlight);
    stats_.gpuTimingAvailable = gpuProfiler_.isAvailable();
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 32;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 32;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(info_.device, &poolInfo, nullptr, &imguiPool_) != VK_SUCCESS)
        return false;
    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = info_.instance;
    initInfo.PhysicalDevice = info_.physicalDevice;
    initInfo.Device = info_.device;
    initInfo.QueueFamily = info_.queueFamily;
    initInfo.Queue = info_.graphicsQueue;
    initInfo.DescriptorPool = imguiPool_;
    initInfo.RenderPass = info_.renderPass;
    initInfo.MinImageCount = info_.imageCount;
    initInfo.ImageCount = info_.imageCount;
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.Allocator = nullptr;
    initInfo.CheckVkResultFn = nullptr;
    if (!ImGui_ImplVulkan_Init(&initInfo))
        return false;
    initialized_ = true;
    imguiReady_ = true;
    return true;
}

void InteractiveChapterTools::shutdown(VkDevice device)
{
    if (!initialized_)
        return;
    vkDeviceWaitIdle(device);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    gpuProfiler_.shutdown(device);
    if (imguiPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, imguiPool_, nullptr);
        imguiPool_ = VK_NULL_HANDLE;
    }
    initialized_ = false;
    imguiReady_ = false;
    if (gTools == this)
        gTools = nullptr;
}

void InteractiveChapterTools::onSwapchainRecreated(
    VkRenderPass renderPass, VkFormat format, uint32_t imageCount)
{
    if (!initialized_)
        return;
    info_.renderPass = renderPass;
    info_.swapchainFormat = format;
    info_.imageCount = imageCount;
    ImGui_ImplVulkan_SetMinImageCount(imageCount);
}

void InteractiveChapterTools::beginFrame(float deltaSeconds)
{
    lastDeltaSeconds_ = deltaSeconds;
    frameTimer_.beginFrame();
    if (info_.window)
        camera_.processKeyboard(info_.window, deltaSeconds);
    if (imguiReady_) {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }
}

void InteractiveChapterTools::buildDebugPanel(const char* chapterTitle)
{
    if (!imguiReady_)
        return;
    ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330.0f, 0.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("调试面板", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextColored(ImVec4(1,0.85f,0.3f,1), "%s", chapterTitle);
        ImGui::Separator();
        ImGui::Text("CPU FPS: %.1f  |  帧时间: %.2f ms", stats_.cpuFps, stats_.cpuFrameMs);
        if (stats_.gpuTimingAvailable)
            ImGui::Text("GPU 帧时间: %.2f ms", stats_.gpuFrameMs);
        else
            ImGui::TextDisabled("GPU 时间戳不可用");
        ImGui::Separator();
        const glm::vec3 eye = camera_.eyePosition();
        glm::vec3 target = camera_.target();
        float distance = camera_.distance();
        ImGui::Text("相机");
        if (ImGui::SliderFloat("距离", &distance, 0.5f, 50.0f))
            camera_.setDistance(distance);
        if (ImGui::DragFloat3("目标点", &target.x, 0.05f))
            camera_.setTarget(target);
        ImGui::Text("眼睛 (%.2f, %.2f, %.2f)", eye.x, eye.y, eye.z);
        ImGui::Text("偏航 %.1f°  俯仰 %.1f°", camera_.yaw(), camera_.pitch());
        ImGui::Separator();
        ImGui::TextDisabled("LMB 旋转 | RMB 平移 | 滚轮 缩放");
        ImGui::TextDisabled("WASD/QE 移动目标 | R 重置 | ESC 退出");
    }
    ImGui::End();
}

void InteractiveChapterTools::beginGpuSection(VkCommandBuffer cmd, uint32_t frameIndex)
{
    if (!gpuProfiler_.isAvailable())
        return;
    gpuProfiler_.reset(cmd, frameIndex);
    gpuProfiler_.writeStart(cmd, frameIndex);
}

void InteractiveChapterTools::endGpuSection(VkCommandBuffer cmd, uint32_t frameIndex)
{
    if (!gpuProfiler_.isAvailable())
        return;
    gpuProfiler_.writeEnd(cmd, frameIndex);
}

void InteractiveChapterTools::renderUi(VkCommandBuffer cmd)
{
    if (!imguiReady_)
        return;
    buildDebugPanel(windowTitleBase_.empty() ? "Chapter" : windowTitleBase_.c_str());
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void InteractiveChapterTools::endFrame(uint32_t frameIndex)
{
    frameTimer_.endFrame();
    stats_.cpuFrameMs = frameTimer_.cpuFrameMs();
    stats_.cpuFps = frameTimer_.cpuFps();
    if (gpuProfiler_.isAvailable() && info_.device != VK_NULL_HANDLE)
        stats_.gpuFrameMs = gpuProfiler_.readGpuFrameMs(info_.device, frameIndex);
    updateWindowTitle();
}

void InteractiveChapterTools::updateWindowTitle() const
{
    if (!info_.window || windowTitleBase_.empty())
        return;
    static thread_local std::string title;
    title = windowTitleBase_ + " | " + formatFrameStats(stats_);
    glfwSetWindowTitle(info_.window, title.c_str());
}

bool InteractiveChapterTools::wantsCaptureMouse() const
{
    return imguiReady_ && ImGui::GetIO().WantCaptureMouse;
}

bool InteractiveChapterTools::wantsCaptureKeyboard() const
{
    return imguiReady_ && ImGui::GetIO().WantCaptureKeyboard;
}

} // namespace vulkan_tutorial
