/**
 * @file ch49_taa.cpp
 * @brief 第49章：时间抗锯齿（Temporal Anti-Aliasing, TAA）
 *
 * 管线：
 *   Pass 1 — 抖动投影渲染场景到 current 颜色纹理
 *   Pass 2 — 与 history 混合（邻域钳制）解析到交换链，并更新 history
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan_tutorial/features.hpp>
#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>
#include <vulkan_tutorial/interactive_chapter.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int      MAX_FRAMES = 2;
constexpr float    TAA_BLEND = 0.9f;

struct TaaVertex {
    glm::vec3 pos;
    glm::vec3 normal;
};

struct SceneUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::mat4 prevViewProj;
    alignas(16) glm::vec2 jitter;
    alignas(16) glm::vec2 resolution;
};

struct ResolvePushConstants {
    float blendFactor;
    glm::vec2 invResolution;
};

static const std::vector<TaaVertex> SCENE_VERTICES = {
    {{-0.8f, -0.5f, 0.0f}, {0, 0, -1}},
    {{ 0.8f, -0.5f, 0.0f}, {0, 0, -1}},
    {{ 0.8f,  0.8f, 0.0f}, {0, 0, -1}},
    {{-0.8f, -0.5f, 0.0f}, {0, 0, -1}},
    {{ 0.8f,  0.8f, 0.0f}, {0, 0, -1}},
    {{-0.8f,  0.8f, 0.0f}, {0, 0, -1}},
    {{-0.3f, -0.2f, 0.5f}, {0, 1, 0}},
    {{ 0.3f, -0.2f, 0.5f}, {0, 1, 0}},
    {{ 0.0f,  0.5f, 0.5f}, {0, 1, 0}},
};

class Ch49App {
public:
    void run() { initWindow(); initVulkan(); mainLoop(); cleanup(); }

private:
    GLFWwindow*      window_         = nullptr;
    VkInstance       instance_       = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_        = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          graphicsQueue_  = VK_NULL_HANDLE;
    VkQueue          presentQueue_   = VK_NULL_HANDLE;
    VkSwapchainKHR   swapchain_      = VK_NULL_HANDLE;
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;
    QueueFamilyIndices queueIndices_{};
    VkFormat         swapchainFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D       swapchainExtent_{};
    DepthResources   depth_{};

    VkImage        currentImage_  = VK_NULL_HANDLE;
    VkDeviceMemory currentMemory_ = VK_NULL_HANDLE;
    VkImageView    currentView_   = VK_NULL_HANDLE;
    VkImage        historyImage_  = VK_NULL_HANDLE;
    VkDeviceMemory historyMemory_ = VK_NULL_HANDLE;
    VkImageView    historyView_   = VK_NULL_HANDLE;
    VkSampler      sampler_       = VK_NULL_HANDLE;

    VkRenderPass   sceneRenderPass_    = VK_NULL_HANDLE;
    VkRenderPass   resolveRenderPass_  = VK_NULL_HANDLE;
    VkFramebuffer  sceneFramebuffer_   = VK_NULL_HANDLE;

    VkDescriptorSetLayout sceneSetLayout_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout resolveSetLayout_  = VK_NULL_HANDLE;
    VkPipelineLayout      scenePipeLayout_   = VK_NULL_HANDLE;
    VkPipelineLayout      resolvePipeLayout_ = VK_NULL_HANDLE;
    VkPipeline            scenePipeline_     = VK_NULL_HANDLE;
    VkPipeline            resolvePipeline_   = VK_NULL_HANDLE;

    VkDescriptorPool      descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneDescSets_;
    VkDescriptorSet       resolveDescSet_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> sceneUBOs_;
    std::vector<VkDeviceMemory> sceneUBOMemories_;
    std::vector<void*>    sceneUBOMapped_;

    VkBuffer       vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;

    std::vector<VkImage>         swapchainImages_;
    std::vector<VkImageView>     swapchainImageViews_;
    std::vector<VkFramebuffer>   resolveFramebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<VkSemaphore>     imageAvailableSems_;
    std::vector<VkSemaphore>     renderFinishedSems_;
    std::vector<VkFence>         inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool     resized_      = false;
    InteractiveChapterTools interactive_;
    bool     historyValid_ = false;
    uint32_t frameIndex_   = 0;
    glm::mat4 prevViewProj_{1.0f};

    static glm::vec2 halton(int index, int base)
    {
        glm::vec2 result(0.0f);
        float f = 1.0f;
        int i = index;
        while (i > 0) {
            f /= static_cast<float>(base);
            result.x += f * static_cast<float>(i % base);
            i /= base;
        }
        f = 1.0f;
        i = index;
        while (i > 0) {
            f /= static_cast<float>(base + 1);
            result.y += f * static_cast<float>(i % (base + 1));
            i /= (base + 1);
        }
        return result;
    }

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT,
            "Ch49 - 时间抗锯齿（TAA 抖动 + History 解析）", nullptr, nullptr);
        interactive_.attachInput(window_);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch49App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan()
    {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physicalDevice_);
        createLogicalDevice(physicalDevice_, surface_, device_, graphicsQueue_,
                            presentQueue_, queueIndices_);
        depth_.format = findDepthFormat(physicalDevice_);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physicalDevice_, device_, swapchainExtent_);
        createOffscreenImages();
        createSampler();
        createRenderPasses();
        createFramebuffers();
        createDescriptorLayouts();
        createPipelines();
        createCommandPool();
        createVertexBuffer();
        createSceneUBOs();
        createDescriptorPoolAndSets();
        createCommandBuffers();
        createSyncObjects();
        InteractiveInitInfo ii{};
        ii.window = window_;
        ii.instance = instance_;
        ii.physicalDevice = physicalDevice_;
        ii.device = device_;
        ii.graphicsQueue = graphicsQueue_;
        ii.queueFamily = queueIndices_.graphicsFamily.value();
        ii.renderPass = resolveRenderPass_;
        ii.swapchainFormat = swapchainFormat_;
        ii.imageCount = static_cast<uint32_t>(swapchainImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setTarget(glm::vec3(0.0f, 0.1f, 0.0f));
        interactive_.camera().setDistance(4.0f);
        std::cout << "\n✅ TAA 初始化完成（Halton 抖动 + History 混合）\n";
    }

    void createSwapchain()
    {
        SwapChainSupportDetails support = querySwapChainSupport(physicalDevice_, surface_);
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(support.presentModes);
        swapchainExtent_ = chooseSwapExtent(support.capabilities, window_);
        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 &&
            imageCount > support.capabilities.maxImageCount)
            imageCount = support.capabilities.maxImageCount;
        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_;
        ci.minImageCount = imageCount;
        ci.imageFormat = surfaceFormat.format;
        ci.imageColorSpace = surfaceFormat.colorSpace;
        ci.imageExtent = swapchainExtent_;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        const uint32_t queueFamilies[] = {
            queueIndices_.graphicsFamily.value(), queueIndices_.presentFamily.value()};
        if (queueIndices_.graphicsFamily != queueIndices_.presentFamily) {
            ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices = queueFamilies;
        } else {
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        ci.preTransform = support.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = presentMode;
        ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
        swapchainImages_.resize(imageCount);
        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchainImages_.data());
        swapchainFormat_ = surfaceFormat.format;
    }

    void createImageViews()
    {
        swapchainImageViews_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image = swapchainImages_[i];
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = swapchainFormat_;
            ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]));
        }
    }

    void createOffscreenImages()
    {
        const VkImageUsageFlags usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        createImage(physicalDevice_, device_, swapchainExtent_.width, swapchainExtent_.height,
                    swapchainFormat_, VK_IMAGE_TILING_OPTIMAL, usage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, currentImage_, currentMemory_);
        createImage(physicalDevice_, device_, swapchainExtent_.width, swapchainExtent_.height,
                    swapchainFormat_, VK_IMAGE_TILING_OPTIMAL, usage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, historyImage_, historyMemory_);
        auto makeView = [&](VkImage img) {
            VkImageViewCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image = img;
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = swapchainFormat_;
            ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageView view = VK_NULL_HANDLE;
            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &view));
            return view;
        };
        currentView_ = makeView(currentImage_);
        historyView_ = makeView(historyImage_);
    }

    void createSampler()
    {
        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &sampler_));
    }

    void createRenderPasses()
    {
        VkAttachmentDescription colorAtt{};
        colorAtt.format = swapchainFormat_;
        colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentDescription depthAtt{};
        depthAtt.format = depth_.format;
        depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &colorRef;
        sp.pDepthStencilAttachment = &depthRef;
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        std::array<VkAttachmentDescription, 2> sceneAtts = {colorAtt, depthAtt};
        VkRenderPassCreateInfo sceneRpi{};
        sceneRpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        sceneRpi.attachmentCount = static_cast<uint32_t>(sceneAtts.size());
        sceneRpi.pAttachments = sceneAtts.data();
        sceneRpi.subpassCount = 1;
        sceneRpi.pSubpasses = &sp;
        sceneRpi.dependencyCount = 1;
        sceneRpi.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &sceneRpi, nullptr, &sceneRenderPass_));

        VkAttachmentDescription swapAtt = colorAtt;
        swapAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkSubpassDescription resolveSp{};
        resolveSp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        resolveSp.colorAttachmentCount = 1;
        resolveSp.pColorAttachments = &colorRef;
        VkRenderPassCreateInfo resolveRpi{};
        resolveRpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        resolveRpi.attachmentCount = 1;
        resolveRpi.pAttachments = &swapAtt;
        resolveRpi.subpassCount = 1;
        resolveRpi.pSubpasses = &resolveSp;
        resolveRpi.dependencyCount = 1;
        resolveRpi.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &resolveRpi, nullptr, &resolveRenderPass_));
    }

    void createFramebuffers()
    {
        VkImageView sceneAtt[] = {currentView_, depth_.view};
        VkFramebufferCreateInfo sceneFb{};
        sceneFb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        sceneFb.renderPass = sceneRenderPass_;
        sceneFb.attachmentCount = 2;
        sceneFb.pAttachments = sceneAtt;
        sceneFb.width = swapchainExtent_.width;
        sceneFb.height = swapchainExtent_.height;
        sceneFb.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &sceneFb, nullptr, &sceneFramebuffer_));

        resolveFramebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            VkImageView att[] = {swapchainImageViews_[i]};
            VkFramebufferCreateInfo fb = sceneFb;
            fb.renderPass = resolveRenderPass_;
            fb.attachmentCount = 1;
            fb.pAttachments = att;
            VK_CHECK(vkCreateFramebuffer(device_, &fb, nullptr, &resolveFramebuffers_[i]));
        }
    }

    void createDescriptorLayouts()
    {
        VkDescriptorSetLayoutBinding sceneBinding{};
        sceneBinding.binding = 0;
        sceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sceneBinding.descriptorCount = 1;
        sceneBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo sceneLayout{};
        sceneLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        sceneLayout.bindingCount = 1;
        sceneLayout.pBindings = &sceneBinding;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &sceneLayout, nullptr, &sceneSetLayout_));
        VkPipelineLayoutCreateInfo scenePl{};
        scenePl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        scenePl.setLayoutCount = 1;
        scenePl.pSetLayouts = &sceneSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &scenePl, nullptr, &scenePipeLayout_));

        std::array<VkDescriptorSetLayoutBinding, 2> resolveBindings{};
        for (uint32_t i = 0; i < 2; ++i) {
            resolveBindings[i].binding = i;
            resolveBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            resolveBindings[i].descriptorCount = 1;
            resolveBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo resolveLayout{};
        resolveLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        resolveLayout.bindingCount = static_cast<uint32_t>(resolveBindings.size());
        resolveLayout.pBindings = resolveBindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &resolveLayout, nullptr, &resolveSetLayout_));
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(ResolvePushConstants);
        VkPipelineLayoutCreateInfo resolvePl{};
        resolvePl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        resolvePl.setLayoutCount = 1;
        resolvePl.pSetLayouts = &resolveSetLayout_;
        resolvePl.pushConstantRangeCount = 1;
        resolvePl.pPushConstantRanges = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &resolvePl, nullptr, &resolvePipeLayout_));
    }

    void createPipelines()
    {
        VkShaderModule sceneVert = createShaderModuleFromFile(device_, "taa.vert.spv");
        VkShaderModule sceneFrag = createShaderModuleFromFile(device_, "taa_scene.frag.spv");
        VkPipelineShaderStageCreateInfo sceneStages[2] = {};
        sceneStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                          VK_SHADER_STAGE_VERTEX_BIT, sceneVert, "main", nullptr};
        sceneStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                          VK_SHADER_STAGE_FRAGMENT_BIT, sceneFrag, "main", nullptr};
        VkVertexInputBindingDescription bind{0, sizeof(TaaVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 2> attrs{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(TaaVertex, pos)};
        attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(TaaVertex, normal)};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
        vi.pVertexAttributeDescriptions = attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};
        dynS.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynS.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        dynS.pDynamicStates = dyn.data();
        VkGraphicsPipelineCreateInfo scenePi{};
        scenePi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        scenePi.stageCount = 2;
        scenePi.pStages = sceneStages;
        scenePi.pVertexInputState = &vi;
        scenePi.pInputAssemblyState = &ia;
        scenePi.pViewportState = &vs;
        scenePi.pRasterizationState = &rs;
        scenePi.pMultisampleState = &ms;
        scenePi.pDepthStencilState = &ds;
        scenePi.pColorBlendState = &cb;
        scenePi.pDynamicState = &dynS;
        scenePi.layout = scenePipeLayout_;
        scenePi.renderPass = sceneRenderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &scenePi, nullptr,
                                           &scenePipeline_));
        vkDestroyShaderModule(device_, sceneFrag, nullptr);
        vkDestroyShaderModule(device_, sceneVert, nullptr);

        VkShaderModule resolveVert = createShaderModuleFromFile(device_, "tonemap.vert.spv");
        VkShaderModule resolveFrag = createShaderModuleFromFile(device_, "taa_resolve.frag.spv");
        VkPipelineShaderStageCreateInfo resolveStages[2] = {};
        resolveStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                            VK_SHADER_STAGE_VERTEX_BIT, resolveVert, "main", nullptr};
        resolveStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                            VK_SHADER_STAGE_FRAGMENT_BIT, resolveFrag, "main", nullptr};
        VkPipelineVertexInputStateCreateInfo emptyVi{};
        emptyVi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkGraphicsPipelineCreateInfo resolvePi = scenePi;
        resolvePi.pStages = resolveStages;
        resolvePi.pVertexInputState = &emptyVi;
        resolvePi.pDepthStencilState = nullptr;
        resolvePi.layout = resolvePipeLayout_;
        resolvePi.renderPass = resolveRenderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &resolvePi, nullptr,
                                           &resolvePipeline_));
        vkDestroyShaderModule(device_, resolveFrag, nullptr);
        vkDestroyShaderModule(device_, resolveVert, nullptr);
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
    }

    void createVertexBuffer()
    {
        const VkDeviceSize size = sizeof(TaaVertex) * SCENE_VERTICES.size();
        createBuffer(physicalDevice_, device_, size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_, vertexMemory_);
        void* mapped = nullptr;
        VK_CHECK(vkMapMemory(device_, vertexMemory_, 0, size, 0, &mapped));
        std::memcpy(mapped, SCENE_VERTICES.data(), static_cast<size_t>(size));
        vkUnmapMemory(device_, vertexMemory_);
    }

    void createSceneUBOs()
    {
        sceneUBOs_.resize(MAX_FRAMES);
        sceneUBOMemories_.resize(MAX_FRAMES);
        sceneUBOMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physicalDevice_, device_, sizeof(SceneUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         sceneUBOs_[i], sceneUBOMemories_[i]);
            VK_CHECK(vkMapMemory(device_, sceneUBOMemories_[i], 0, sizeof(SceneUBO), 0,
                                 &sceneUBOMapped_[i]));
        }
    }

    void createDescriptorPoolAndSets()
    {
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(MAX_FRAMES)},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        };
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.poolSizeCount = 2;
        pool.pPoolSizes = sizes;
        pool.maxSets = static_cast<uint32_t>(MAX_FRAMES) + 1;
        VK_CHECK(vkCreateDescriptorPool(device_, &pool, nullptr, &descriptorPool_));
        sceneDescSets_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptorPool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &sceneSetLayout_;
            VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &sceneDescSets_[i]));
            VkDescriptorBufferInfo bufInfo{sceneUBOs_[i], 0, sizeof(SceneUBO)};
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = sceneDescSets_[i];
            write.dstBinding = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &bufInfo;
            vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        }
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &resolveSetLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &resolveDescSet_));
        VkDescriptorImageInfo images[2] = {
            {sampler_, currentView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {sampler_, historyView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        };
        VkWriteDescriptorSet writes[2] = {};
        for (uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = resolveDescSet_;
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].descriptorCount = 1;
            writes[i].pImageInfo = &images[i];
        }
        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    }

    void createCommandBuffers()
    {
        commandBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()));
    }

    void createSyncObjects()
    {
        imageAvailableSems_.resize(MAX_FRAMES);
        renderFinishedSems_.resize(MAX_FRAMES);
        inFlightFences_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo sem{};
        sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &sem, nullptr, &imageAvailableSems_[i]));
            VK_CHECK(vkCreateSemaphore(device_, &sem, nullptr, &renderFinishedSems_[i]));
            VK_CHECK(vkCreateFence(device_, &fence, nullptr, &inFlightFences_[i]));
        }
    }

    void updateUBO(uint32_t frameIndex, float time)
    {
        SceneUBO ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * 0.6f, glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.view = interactive_.camera().viewMatrix();
        const float aspect = static_cast<float>(swapchainExtent_.width) /
                             static_cast<float>(swapchainExtent_.height);
        glm::mat4 proj = interactive_.camera().projectionMatrix(aspect, 45.0f, 0.1f, 20.0f);
        const glm::vec2 haltonSample = halton(static_cast<int>((frameIndex_ % 16) + 1), 2);
        const glm::vec2 pixelJitter = (haltonSample - glm::vec2(0.5f)) *
            glm::vec2(2.0f / static_cast<float>(swapchainExtent_.width),
                      2.0f / static_cast<float>(swapchainExtent_.height));
        proj[2][0] += pixelJitter.x;
        proj[2][1] += pixelJitter.y;
        ubo.projection = proj;
        ubo.prevViewProj = prevViewProj_;
        ubo.jitter = pixelJitter;
        ubo.resolution = glm::vec2(static_cast<float>(swapchainExtent_.width),
                                     static_cast<float>(swapchainExtent_.height));
        prevViewProj_ = proj * ubo.view * ubo.model;
        std::memcpy(sceneUBOMapped_[frameIndex], &ubo, sizeof(ubo));
    }

    void copyImageToHistory(VkCommandBuffer cmd)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.image = currentImage_;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.image = historyImage_;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.extent = {swapchainExtent_.width, swapchainExtent_.height, 1};
        vkCmdCopyImage(cmd, currentImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       historyImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.image = historyImage_;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.image = currentImage_;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex, float time)
    {
        updateUBO(currentFrame_, time);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, currentFrame_);
        VkViewport vp{0.0f, 0.0f, static_cast<float>(swapchainExtent_.width),
                      static_cast<float>(swapchainExtent_.height), 0.0f, 1.0f};
        VkRect2D sc{{0, 0}, swapchainExtent_};
        std::array<VkClearValue, 2> clears{};
        clears[0] = {{{0.05f, 0.05f, 0.08f, 1.0f}}};
        clears[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo sceneRp{};
        sceneRp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        sceneRp.renderPass = sceneRenderPass_;
        sceneRp.framebuffer = sceneFramebuffer_;
        sceneRp.renderArea = sc;
        sceneRp.clearValueCount = static_cast<uint32_t>(clears.size());
        sceneRp.pClearValues = clears.data();
        vkCmdBeginRenderPass(cmd, &sceneRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeLayout_, 0, 1,
                                &sceneDescSets_[currentFrame_], 0, nullptr);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer_, &offset);
        vkCmdDraw(cmd, static_cast<uint32_t>(SCENE_VERTICES.size()), 1, 0, 0);
        vkCmdEndRenderPass(cmd);

        ResolvePushConstants pc{};
        pc.blendFactor = historyValid_ ? TAA_BLEND : 0.0f;
        pc.invResolution = glm::vec2(1.0f / static_cast<float>(swapchainExtent_.width),
                                     1.0f / static_cast<float>(swapchainExtent_.height));
        VkClearValue swapClear = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        VkRenderPassBeginInfo resolveRp{};
        resolveRp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        resolveRp.renderPass = resolveRenderPass_;
        resolveRp.framebuffer = resolveFramebuffers_[imageIndex];
        resolveRp.renderArea = sc;
        resolveRp.clearValueCount = 1;
        resolveRp.pClearValues = &swapClear;
        vkCmdBeginRenderPass(cmd, &resolveRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, resolvePipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, resolvePipeLayout_, 0, 1,
                                &resolveDescSet_, 0, nullptr);
        vkCmdPushConstants(cmd, resolvePipeLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(pc), &pc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);
        copyImageToHistory(cmd);
        historyValid_ = true;
        ++frameIndex_;
        interactive_.endGpuSection(cmd, currentFrame_);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame()
    {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex = 0;
        VkResult result = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
            imageAvailableSems_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("获取交换链图像失败");
        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        static auto start = std::chrono::steady_clock::now();
        float time = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
        recordCommandBuffer(commandBuffers_[currentFrame_], imageIndex, time);
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        VkSemaphore waitSems[] = {imageAvailableSems_[currentFrame_]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = waitSems;
        submit.pWaitDstStageMask = waitStages;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffers_[currentFrame_];
        VkSemaphore signalSems[] = {renderFinishedSems_[currentFrame_]};
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = signalSems;
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &submit, inFlightFences_[currentFrame_]));
        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = signalSems;
        VkSwapchainKHR chains[] = {swapchain_};
        present.swapchainCount = 1;
        present.pSwapchains = chains;
        present.pImageIndices = &imageIndex;
        result = vkQueuePresentKHR(presentQueue_, &present);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("呈现失败");
        }
        interactive_.endFrame(currentFrame_);
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void cleanupOffscreen()
    {
        if (currentView_) vkDestroyImageView(device_, currentView_, nullptr);
        if (historyView_) vkDestroyImageView(device_, historyView_, nullptr);
        if (currentImage_) vkDestroyImage(device_, currentImage_, nullptr);
        if (historyImage_) vkDestroyImage(device_, historyImage_, nullptr);
        if (currentMemory_) vkFreeMemory(device_, currentMemory_, nullptr);
        if (historyMemory_) vkFreeMemory(device_, historyMemory_, nullptr);
        currentView_ = historyView_ = VK_NULL_HANDLE;
        currentImage_ = historyImage_ = VK_NULL_HANDLE;
        currentMemory_ = historyMemory_ = VK_NULL_HANDLE;
    }

    void recreateSwapchain()
    {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        for (auto fb : resolveFramebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto iv : swapchainImageViews_) vkDestroyImageView(device_, iv, nullptr);
        vkDestroyFramebuffer(device_, sceneFramebuffer_, nullptr);
        destroyDepthResources(device_, depth_);
        cleanupOffscreen();
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        historyValid_ = false;
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physicalDevice_, device_, swapchainExtent_);
        createOffscreenImages();
        createFramebuffers();
        VkDescriptorImageInfo images[2] = {
            {sampler_, currentView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {sampler_, historyView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        };
        VkWriteDescriptorSet writes[2] = {};
        for (uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = resolveDescSet_;
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].descriptorCount = 1;
            writes[i].pImageInfo = &images[i];
        }
        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
        interactive_.onSwapchainRecreated(resolveRenderPass_, swapchainFormat_,
            static_cast<uint32_t>(swapchainImages_.size()));
    }

    void mainLoop()
    {
        std::cout << "🎨 旋转几何体 + TAA 抖动。ESC 退出。\n";
        auto lastTime = std::chrono::steady_clock::now();
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            const auto now = std::chrono::steady_clock::now();
            const float dt = std::chrono::duration<float>(now - lastTime).count();
            lastTime = now;
            interactive_.beginFrame(dt);
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    void cleanup()
    {
        vkDestroyPipeline(device_, resolvePipeline_, nullptr);
        vkDestroyPipeline(device_, scenePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, resolvePipeLayout_, nullptr);
        vkDestroyPipelineLayout(device_, scenePipeLayout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, resolveSetLayout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, sceneSetLayout_, nullptr);
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkUnmapMemory(device_, sceneUBOMemories_[i]);
            vkDestroyBuffer(device_, sceneUBOs_[i], nullptr);
            vkFreeMemory(device_, sceneUBOMemories_[i], nullptr);
        }
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMemory_, nullptr);
        vkDestroySampler(device_, sampler_, nullptr);
        cleanupOffscreen();
        destroyDepthResources(device_, depth_);
        for (auto fb : resolveFramebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyFramebuffer(device_, sceneFramebuffer_, nullptr);
        for (auto iv : swapchainImageViews_) vkDestroyImageView(device_, iv, nullptr);
        vkDestroyRenderPass(device_, resolveRenderPass_, nullptr);
        vkDestroyRenderPass(device_, sceneRenderPass_, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        interactive_.shutdown(device_);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
};

int main()
{
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << " 第49章：Temporal Anti-Aliasing\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    Ch49App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
