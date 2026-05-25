/**
 * @file mini_engine.cpp
 * @brief 第70章：MiniEngine 主类实现
 *
 * 整合所有引擎子系统：
 *   RHIDevice / TextureCache / ShaderLibrary / MaterialLibrary
 *   PipelineCache / CommandPool / World / FrustumCuller / DrawCallBatch
 *
 * 与 ch60（~1600 行裸 Vulkan）对比：
 *   使用 MiniEngine 的等价场景 Demo 仅需约 150 行。
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan_tutorial/engine/mini_engine.hpp>
#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>

#include <algorithm>

using namespace vulkan_tutorial;
#include <chrono>
#include <stdexcept>

namespace engine {

// ─── 辅助：单实例 GLFW 回调转发 ──────────────────────────────────────────────

static void glfwKeyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    (void)window;
    (void)key;
    (void)action;
}

static void glfwResizeCallback(GLFWwindow* window, int /*w*/, int /*h*/)
{
    MiniEngine* eng = reinterpret_cast<MiniEngine*>(glfwGetWindowUserPointer(window));
    if (eng) eng->markResized();
}

// ─── MiniEngine::run ─────────────────────────────────────────────────────────

void MiniEngine::run(const EngineConfig& config, Application& app)
{
    init(config);
    app.onInit(*this);
    mainLoop(app);
    device_.waitIdle();
    app.onShutdown(*this);
    shutdown();
}

// ─── MiniEngine::init ────────────────────────────────────────────────────────

void MiniEngine::init(const EngineConfig& config)
{
    config_ = config;

    // GLFW 初始化 + 窗口创建
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(
        static_cast<int>(config_.width),
        static_cast<int>(config_.height),
        config_.appName.c_str(),
        nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, glfwKeyCallback);
    glfwSetFramebufferSizeCallback(window_, glfwResizeCallback);

    // RHIDevice：实例 + 物理设备 + 逻辑设备
    DeviceCreateInfo dci{};
    dci.windowHandle      = window_;
    dci.enableValidation  = config_.enableValidation;
    dci.preferDiscreteGpu = true;
    dci.frameCount        = config_.frameCount;
    device_.init(dci);

    // 子系统初始化
    textures_.init(device_);
    shaders_.init(device_, config_.shaderDir);
    materials_.init(device_, config_.frameCount);
    pipelineCache_.init(device_);

    // 命令池：每帧一个
    cmdPool_.create(device_, config_.frameCount);
    cmdBufs_.resize(config_.frameCount);
    for (uint32_t i = 0; i < config_.frameCount; ++i)
        cmdBufs_[i] = cmdPool_.allocate(i);

    createSwapchain();
    createRenderPass();
    createFramebuffers();
    createSyncObjects();
}

// ─── MiniEngine::mainLoop ────────────────────────────────────────────────────

void MiniEngine::mainLoop(Application& app)
{
    auto lastTime = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        totalTime_ += dt;

        renderFrame(app, dt);
    }
}

// ─── MiniEngine::renderFrame ─────────────────────────────────────────────────

void MiniEngine::renderFrame(Application& app, float dt)
{
    uint32_t fi = currentFrame_;
    VkDevice dev = device_.device();

    VK_CHECK(vkWaitForFences(dev, 1, &inFlight_[fi], VK_TRUE, UINT64_MAX));

    uint32_t imageIndex = 0;
    VkResult acqResult = vkAcquireNextImageKHR(dev, swapchain_, UINT64_MAX,
        imgAvail_[fi], VK_NULL_HANDLE, &imageIndex);

    if (acqResult == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    if (acqResult != VK_SUCCESS && acqResult != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("vkAcquireNextImageKHR 失败");

    VK_CHECK(vkResetFences(dev, 1, &inFlight_[fi]));

    // 更新逻辑
    app.onUpdate(*this, dt);
    transformSystem(world_);

    // 视锥裁剪
    std::vector<EntityID> visibleEntities;
    EntityID camEntity = world_.findByTag("MainCamera");
    if (camEntity != NULL_ENTITY) {
        TransformComponent* camTc = world_.get<TransformComponent>(camEntity);
        CameraComponent* camComp  = world_.get<CameraComponent>(camEntity);
        if (camTc && camComp) {
            float aspect = static_cast<float>(extent_.width) /
                           static_cast<float>(extent_.height);
            glm::mat4 view = glm::inverse(camTc->world);
            glm::mat4 proj = camComp->projMatrix(aspect);
            Frustum frustum = Frustum::fromViewProj(proj * view);
            culler_.cull(world_, frustum, visibleEntities);
        }
    }

    // 录制命令
    cmdPool_.reset(fi);
    VkCommandBuffer cmd = cmdBufs_[fi];

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    // 设置 viewport / scissor
    VkViewport vp{};
    vp.width    = static_cast<float>(extent_.width);
    vp.height   = static_cast<float>(extent_.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{ 0, 0 }, extent_};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // 清值
    VkClearValue colorClear{};
    colorClear.color.float32[0] = 0.05f;
    colorClear.color.float32[1] = 0.05f;
    colorClear.color.float32[2] = 0.1f;
    colorClear.color.float32[3] = 1.0f;
    VkClearValue depthClear{};
    depthClear.depthStencil.depth = 1.0f;
    depthClear.depthStencil.stencil = 0;

    std::array<VkClearValue, 2> clearValues = { colorClear, depthClear };

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass        = mainRP_;
    rpbi.framebuffer       = swapFBs_[imageIndex];
    rpbi.renderArea.extent = extent_;
    rpbi.clearValueCount   = static_cast<uint32_t>(clearValues.size());
    rpbi.pClearValues      = clearValues.data();
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    // 更新材质描述符
    materials_.updateAll(fi);

    // 用户渲染回调
    FrameContext ctx{};
    ctx.cmd        = cmd;
    ctx.frameIndex = fi;
    ctx.imageIndex = imageIndex;
    ctx.deltaTime  = dt;
    ctx.totalTime  = totalTime_;
    ctx.extent     = extent_;
    app.onRender(*this, ctx);

    // 排序并提交 DrawCallBatch
    drawBatch_.sort();
    drawBatch_.flush(cmd);
    drawBatch_.clear();

    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    // 提交
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &imgAvail_[fi];
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &renderDone_[fi];
    VK_CHECK(vkQueueSubmit(device_.graphicsQueue().handle, 1, &si, inFlight_[fi]));

    // 呈现
    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &renderDone_[fi];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain_;
    pi.pImageIndices      = &imageIndex;
    VkResult presResult = vkQueuePresentKHR(device_.graphicsQueue().handle, &pi);

    if (presResult == VK_ERROR_OUT_OF_DATE_KHR || presResult == VK_SUBOPTIMAL_KHR || resized_) {
        resized_ = false;
        recreateSwapchain();
    } else if (presResult != VK_SUCCESS) {
        throw std::runtime_error("vkQueuePresentKHR 失败");
    }

    currentFrame_ = (currentFrame_ + 1) % config_.frameCount;
}

// ─── MiniEngine::createSwapchain ─────────────────────────────────────────────

void MiniEngine::createSwapchain()
{
    VkPhysicalDevice physDev = device_.physicalDevice();
    VkDevice         dev     = device_.device();
    VkSurfaceKHR     surface = device_.surface();

    SwapChainSupportDetails support =
        querySwapChainSupport(physDev, surface);
    VkSurfaceFormatKHR fmt  = chooseSwapSurfaceFormat(support.formats);
    VkPresentModeKHR   mode = config_.vsync
        ? VK_PRESENT_MODE_FIFO_KHR
        : chooseSwapPresentMode(support.presentModes);
    extent_ = chooseSwapExtent(support.capabilities, window_);

    uint32_t imgCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 &&
        imgCount > support.capabilities.maxImageCount)
        imgCount = support.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = surface;
    ci.minImageCount    = imgCount;
    ci.imageFormat      = fmt.format;
    ci.imageColorSpace  = fmt.colorSpace;
    ci.imageExtent      = extent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform     = support.capabilities.currentTransform;
    ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode      = mode;
    ci.clipped          = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(dev, &ci, nullptr, &swapchain_));
    swapFormat_ = fmt.format;

    uint32_t count = 0;
    vkGetSwapchainImagesKHR(dev, swapchain_, &count, nullptr);
    swapImages_.resize(count);
    vkGetSwapchainImagesKHR(dev, swapchain_, &count, swapImages_.data());

    swapViews_.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        VkImageViewCreateInfo vci{};
        vci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image                           = swapImages_[i];
        vci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        vci.format                          = swapFormat_;
        vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel   = 0;
        vci.subresourceRange.levelCount     = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(dev, &vci, nullptr, &swapViews_[i]));
    }

    // 深度图
    depth_ = createDepthResources(physDev, dev, extent_);
}

// ─── MiniEngine::createRenderPass ────────────────────────────────────────────

void MiniEngine::createRenderPass()
{
    VkAttachmentDescription color{};
    color.format         = swapFormat_;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth{};
    depth.format         = depth_.format != VK_FORMAT_UNDEFINED
                         ? depth_.format
                         : device_.depthFormat();
    depth.samples        = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                      | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                      | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                      | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = { color, depth };
    VkRenderPassCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = static_cast<uint32_t>(attachments.size());
    ci.pAttachments    = attachments.data();
    ci.subpassCount    = 1;
    ci.pSubpasses      = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies   = &dep;

    VK_CHECK(vkCreateRenderPass(device_.device(), &ci, nullptr, &mainRP_));
}

// ─── MiniEngine::createFramebuffers ──────────────────────────────────────────

void MiniEngine::createFramebuffers()
{
    swapFBs_.resize(swapViews_.size());
    for (size_t i = 0; i < swapViews_.size(); ++i) {
        std::array<VkImageView, 2> attachments = { swapViews_[i], depth_.view };
        VkFramebufferCreateInfo ci{};
        ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass      = mainRP_;
        ci.attachmentCount = static_cast<uint32_t>(attachments.size());
        ci.pAttachments    = attachments.data();
        ci.width           = extent_.width;
        ci.height          = extent_.height;
        ci.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(device_.device(), &ci, nullptr, &swapFBs_[i]));
    }
}

// ─── MiniEngine::createSyncObjects ───────────────────────────────────────────

void MiniEngine::createSyncObjects()
{
    imgAvail_.resize(config_.frameCount);
    renderDone_.resize(config_.frameCount);
    inFlight_.resize(config_.frameCount);

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < config_.frameCount; ++i) {
        VK_CHECK(vkCreateSemaphore(device_.device(), &sci, nullptr, &imgAvail_[i]));
        VK_CHECK(vkCreateSemaphore(device_.device(), &sci, nullptr, &renderDone_[i]));
        VK_CHECK(vkCreateFence   (device_.device(), &fci, nullptr, &inFlight_[i]));
    }
}

// ─── MiniEngine::recreateSwapchain ───────────────────────────────────────────

void MiniEngine::recreateSwapchain()
{
    int w = 0, h = 0;
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(window_, &w, &h);
        glfwWaitEvents();
    }
    device_.waitIdle();

    // 销毁旧资源
    for (auto fb : swapFBs_)
        vkDestroyFramebuffer(device_.device(), fb, nullptr);
    swapFBs_.clear();

    if (mainRP_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device_.device(), mainRP_, nullptr);
        mainRP_ = VK_NULL_HANDLE;
    }

    for (auto view : swapViews_)
        vkDestroyImageView(device_.device(), view, nullptr);
    swapViews_.clear();

    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_.device(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }

    destroyDepthResources(device_.device(), depth_);

    createSwapchain();
    createRenderPass();
    createFramebuffers();
}

// ─── MiniEngine::submit ──────────────────────────────────────────────────────

void MiniEngine::submit(EntityID entity,
                         VkPipeline pipeline,
                         VkDescriptorSet descSet,
                         DrawKey::Layer layer)
{
    MeshComponent* mesh = world_.get<MeshComponent>(entity);
    if (!mesh) return;

    DrawCall dc{};
    dc.key        = DrawKey::make(layer, 0, 0, entity & 0x00FFFFFFu);
    dc.pipeline   = pipeline;
    dc.descSet    = descSet;
    dc.vertexBuf  = mesh->vertexBuffer;
    dc.indexBuf   = mesh->indexBuffer;
    dc.indexCount = mesh->indexCount;
    drawBatch_.add(std::move(dc));
}

// ─── MiniEngine::createFramebuffer ───────────────────────────────────────────

VkFramebuffer MiniEngine::createFramebuffer(VkRenderPass rp,
                                              const std::vector<VkImageView>& views)
{
    VkFramebufferCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    ci.renderPass      = rp;
    ci.attachmentCount = static_cast<uint32_t>(views.size());
    ci.pAttachments    = views.data();
    ci.width           = extent_.width;
    ci.height          = extent_.height;
    ci.layers          = 1;

    VkFramebuffer fb = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFramebuffer(device_.device(), &ci, nullptr, &fb));
    return fb;
}

// ─── MiniEngine::setupForwardPass ────────────────────────────────────────────

void MiniEngine::setupForwardPass(VkFormat /*colorFmt*/, bool /*withDepth*/,
                                    bool /*withBloom*/)
{
    // 默认 forward pass 已在 createRenderPass() 中创建
    // 此方法保留供将来扩展（多 pass、Bloom 等）
}

// ─── MiniEngine::shutdown ────────────────────────────────────────────────────

void MiniEngine::shutdown()
{
    if (!device_.isValid()) return;
    device_.waitIdle();

    materials_.destroy();
    pipelineCache_.destroy();
    shaders_.destroy();
    textures_.destroy();

    for (auto fb : swapFBs_)
        vkDestroyFramebuffer(device_.device(), fb, nullptr);

    if (mainRP_ != VK_NULL_HANDLE)
        vkDestroyRenderPass(device_.device(), mainRP_, nullptr);

    for (auto view : swapViews_)
        vkDestroyImageView(device_.device(), view, nullptr);

    destroyDepthResources(device_.device(), depth_);

    if (swapchain_ != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(device_.device(), swapchain_, nullptr);

    for (uint32_t i = 0; i < config_.frameCount; ++i) {
        vkDestroySemaphore(device_.device(), imgAvail_[i], nullptr);
        vkDestroySemaphore(device_.device(), renderDone_[i], nullptr);
        vkDestroyFence   (device_.device(), inFlight_[i],   nullptr);
    }

    cmdPool_.destroy();

    if (window_) {
        glfwDestroyWindow(window_);
        glfwTerminate();
        window_ = nullptr;
    }
}

} // namespace engine
