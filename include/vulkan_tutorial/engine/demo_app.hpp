#pragma once
/**
 * @file demo_app.hpp
 * @brief ch62–ch70 Demo 章节共用的最简 Vulkan 应用基类
 *
 * 提供：交换链 + 单色 RenderPass + 帧同步 + ImGui
 * 子类只需实现 buildUi() 填充 ImGui 面板。
 *
 * 使用方法：
 *   class Ch62App : public DemoApp {
 *       void buildUi() override { ... ImGui calls ... }
 *   };
 *   int main() { Ch62App app; app.run("标题"); }
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>
#include <vulkan_tutorial/interactive_chapter.hpp>

#include <imgui.h>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vulkan_tutorial;

constexpr int DEMO_MAX_FRAMES = 2;

class DemoApp {
  public:
    void run(const char* windowTitle, uint32_t w = 960, uint32_t h = 720) {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(int(w), int(h), windowTitle, nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* win, int, int) {
            reinterpret_cast<DemoApp*>(glfwGetWindowUserPointer(win))->resized_ = true;
        });
        interactive_.attachInput(window_);
        setupVulkan();
        onInit();
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            interactive_.beginFrame(0.016f);
            onUpdate();
            buildUi();
            drawFrame();
            interactive_.updateWindowTitle();
        }
        vkDeviceWaitIdle(device_);
        onShutdown();
        teardown();
    }

  protected:
    virtual void buildUi() {}    ///< 子类填充 ImGui
    virtual void onInit() {}     ///< 子类初始化（设备已就绪）
    virtual void onUpdate() {}   ///< 每帧逻辑
    virtual void onShutdown() {} ///< 子类清理

    // 子类可以访问的基础资源
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physDev_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue gQueue_ = VK_NULL_HANDLE;
    VkQueue pQueue_ = VK_NULL_HANDLE;
    QueueFamilyIndices qIdx_{};
    VkFormat swapFmt_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    InteractiveChapterTools interactive_;

    glm::vec3 bgColor_{0.05f, 0.07f, 0.12f}; ///< 背景色（子类可改）

    void waitIdle() const {
        vkDeviceWaitIdle(device_);
    }

  private:
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<VkFramebuffer> swapFBs_;
    std::vector<VkCommandBuffer> cmds_;
    std::vector<VkSemaphore> imgAvail_, renderDone_;
    std::vector<VkFence> inFlight_;
    uint32_t frame_ = 0;
    bool resized_ = false;

    void setupVulkan() {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physDev_);
        createLogicalDevice(physDev_, surface_, device_, gQueue_, pQueue_, qIdx_);

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = qIdx_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &cpci, nullptr, &cmdPool_));

        createSwapchainObjects();

        InteractiveInitInfo ii{};
        ii.window = window_;
        ii.instance = instance_;
        ii.physicalDevice = physDev_;
        ii.device = device_;
        ii.graphicsQueue = gQueue_;
        ii.queueFamily = qIdx_.graphicsFamily.value();
        ii.renderPass = renderPass_;
        ii.swapchainFormat = swapFmt_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = DEMO_MAX_FRAMES;
        interactive_.initVulkan(ii);
    }

    void createSwapchainObjects() {
        auto d = querySwapChainSupport(physDev_, surface_);
        auto f = chooseSwapSurfaceFormat(d.formats);
        auto m = chooseSwapPresentMode(d.presentModes);
        extent_ = chooseSwapExtent(d.capabilities, window_);
        swapFmt_ = f.format;
        uint32_t cnt = d.capabilities.minImageCount + 1;
        if (d.capabilities.maxImageCount)
            cnt = std::min(cnt, d.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        sci.surface = surface_;
        sci.minImageCount = cnt;
        sci.imageFormat = f.format;
        sci.imageColorSpace = f.colorSpace;
        sci.imageExtent = extent_;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        uint32_t qf[2] = {qIdx_.graphicsFamily.value(), qIdx_.presentFamily.value()};
        if (qf[0] != qf[1]) {
            sci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            sci.queueFamilyIndexCount = 2;
            sci.pQueueFamilyIndices = qf;
        } else
            sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = d.capabilities.currentTransform;
        sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        sci.presentMode = m;
        sci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &sci, nullptr, &swapchain_));
        uint32_t n = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
        swapImages_.resize(n);
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, swapImages_.data());

        swapViews_.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = swapImages_[i];
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = swapFmt_;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &swapViews_[i]));
        }

        if (renderPass_ == VK_NULL_HANDLE) {
            VkAttachmentDescription att{};
            att.format = swapFmt_;
            att.samples = VK_SAMPLE_COUNT_1_BIT;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            VkAttachmentReference cr{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1;
            sub.pColorAttachments = &cr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount = 1;
            rpci.pAttachments = &att;
            rpci.subpassCount = 1;
            rpci.pSubpasses = &sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &renderPass_));
        }

        swapFBs_.resize(n);
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = renderPass_;
        fci.attachmentCount = 1;
        fci.width = extent_.width;
        fci.height = extent_.height;
        fci.layers = 1;
        for (uint32_t i = 0; i < n; ++i) {
            fci.pAttachments = &swapViews_[i];
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &swapFBs_[i]));
        }

        if (cmds_.empty()) {
            cmds_.resize(DEMO_MAX_FRAMES);
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = cmdPool_;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = DEMO_MAX_FRAMES;
            VK_CHECK(vkAllocateCommandBuffers(device_, &ai, cmds_.data()));
            imgAvail_.resize(DEMO_MAX_FRAMES);
            renderDone_.resize(DEMO_MAX_FRAMES);
            inFlight_.resize(DEMO_MAX_FRAMES);
            VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            for (int i = 0; i < DEMO_MAX_FRAMES; ++i) {
                VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &imgAvail_[i]));
                VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &renderDone_[i]));
                VK_CHECK(vkCreateFence(device_, &fi, nullptr, &inFlight_[i]));
            }
        }
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlight_[frame_], VK_TRUE, UINT64_MAX);
        uint32_t idx = 0;
        VkResult r = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imgAvail_[frame_], VK_NULL_HANDLE, &idx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            recreate();
            return;
        }
        vkResetFences(device_, 1, &inFlight_[frame_]);

        VkCommandBuffer cmd = cmds_[frame_];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, frame_);

        VkClearValue cv{};
        cv.color.float32[0] = bgColor_.r;
        cv.color.float32[1] = bgColor_.g;
        cv.color.float32[2] = bgColor_.b;
        cv.color.float32[3] = 1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = renderPass_;
        rbi.framebuffer = swapFBs_[idx];
        rbi.renderArea.extent = extent_;
        rbi.clearValueCount = 1;
        rbi.pClearValues = &cv;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);

        interactive_.endGpuSection(cmd, frame_);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &imgAvail_[frame_];
        si.pWaitDstStageMask = &wait;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &renderDone_[frame_];
        VK_CHECK(vkQueueSubmit(gQueue_, 1, &si, inFlight_[frame_]));
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &renderDone_[frame_];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain_;
        pi.pImageIndices = &idx;
        r = vkQueuePresentKHR(pQueue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreate();
        }
        interactive_.endFrame(frame_);
        frame_ = (frame_ + 1) % DEMO_MAX_FRAMES;
    }

    void recreate() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        for (auto fb : swapFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        swapFBs_.clear();
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchainObjects();
        interactive_.onSwapchainRecreated(renderPass_, swapFmt_, static_cast<uint32_t>(swapImages_.size()));
    }

    void teardown() {
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        for (auto fb : swapFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        for (int i = 0; i < DEMO_MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imgAvail_[i], nullptr);
            vkDestroySemaphore(device_, renderDone_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
        interactive_.shutdown(device_);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
};
