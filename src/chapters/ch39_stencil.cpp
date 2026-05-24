/**
 * @file ch39_stencil.cpp
 * @brief 第39章：模板缓冲（Stencil Buffer）描边
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【模板缓冲用途】
 *
 *  模板缓冲与深度缓冲共享同一 VkImage（D24/D32 + S8），
 *  每个像素额外存储 8 位模板值，用于遮罩、描边、阴影体积等。
 *
 * 【两遍描边流程】
 *
 *  Pass 1 — 写入模板（不写颜色）：
 *  ┌─────────────────────────────────────┐
 *  │  正常尺寸物体 → stencil = 1         │
 *  │  colorWriteMask = 0                 │
 *  └─────────────────────────────────────┘
 *
 *  Pass 2 — 放大描边（模板测试）：
 *  ┌─────────────────────────────────────┐
 *  │  放大 1.08x 的物体                  │
 *  │  stencilCompare = NOT_EQUAL (ref=1) │
 *  │  只在 stencil≠1 处绘制 → 描边环     │
 *  └─────────────────────────────────────┘
 *
 *      ████████████   ← 放大后的轮廓（黄色）
 *      ██        ██
 *      ██  物体  ██   ← 内部 stencil=1，被剔除
 *      ██        ██
 *      ████████████
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan_tutorial/features.hpp>
#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>
#include <vulkan_tutorial/interactive_chapter.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH      = 800;
constexpr uint32_t HEIGHT     = 600;
constexpr int      MAX_FRAMES = 2;
constexpr uint32_t STENCIL_REF = 1;

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription d{};
        d.binding   = 0;
        d.stride    = sizeof(Vertex);
        d.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return d;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> a{};
        a[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
        a[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
        return a;
    }
};

// 彩色立方体（36 顶点，每面两三角形）
static const std::vector<Vertex> CUBE_VERTICES = {
    {{-0.5f,-0.5f, 0.5f},{0.8f,0.2f,0.2f}}, {{ 0.5f,-0.5f, 0.5f},{0.8f,0.2f,0.2f}},
    {{ 0.5f, 0.5f, 0.5f},{0.8f,0.2f,0.2f}}, {{-0.5f,-0.5f, 0.5f},{0.8f,0.2f,0.2f}},
    {{ 0.5f, 0.5f, 0.5f},{0.8f,0.2f,0.2f}}, {{-0.5f, 0.5f, 0.5f},{0.8f,0.2f,0.2f}},

    {{ 0.5f,-0.5f, 0.5f},{0.2f,0.8f,0.2f}}, {{ 0.5f,-0.5f,-0.5f},{0.2f,0.8f,0.2f}},
    {{ 0.5f, 0.5f,-0.5f},{0.2f,0.8f,0.2f}}, {{ 0.5f,-0.5f, 0.5f},{0.2f,0.8f,0.2f}},
    {{ 0.5f, 0.5f,-0.5f},{0.2f,0.8f,0.2f}}, {{ 0.5f, 0.5f, 0.5f},{0.2f,0.8f,0.2f}},

    {{ 0.5f,-0.5f,-0.5f},{0.2f,0.2f,0.8f}}, {{-0.5f,-0.5f,-0.5f},{0.2f,0.2f,0.8f}},
    {{-0.5f, 0.5f,-0.5f},{0.2f,0.2f,0.8f}}, {{ 0.5f,-0.5f,-0.5f},{0.2f,0.2f,0.8f}},
    {{-0.5f, 0.5f,-0.5f},{0.2f,0.2f,0.8f}}, {{ 0.5f, 0.5f,-0.5f},{0.2f,0.2f,0.8f}},

    {{-0.5f,-0.5f,-0.5f},{0.8f,0.8f,0.2f}}, {{-0.5f,-0.5f, 0.5f},{0.8f,0.8f,0.2f}},
    {{-0.5f, 0.5f, 0.5f},{0.8f,0.8f,0.2f}}, {{-0.5f,-0.5f,-0.5f},{0.8f,0.8f,0.2f}},
    {{-0.5f, 0.5f, 0.5f},{0.8f,0.8f,0.2f}}, {{-0.5f, 0.5f,-0.5f},{0.8f,0.8f,0.2f}},

    {{-0.5f, 0.5f, 0.5f},{0.8f,0.2f,0.8f}}, {{ 0.5f, 0.5f, 0.5f},{0.8f,0.2f,0.8f}},
    {{ 0.5f, 0.5f,-0.5f},{0.8f,0.2f,0.8f}}, {{-0.5f, 0.5f, 0.5f},{0.8f,0.2f,0.8f}},
    {{ 0.5f, 0.5f,-0.5f},{0.8f,0.2f,0.8f}}, {{-0.5f, 0.5f,-0.5f},{0.8f,0.2f,0.8f}},

    {{-0.5f,-0.5f,-0.5f},{0.2f,0.8f,0.8f}}, {{ 0.5f,-0.5f,-0.5f},{0.2f,0.8f,0.8f}},
    {{ 0.5f,-0.5f, 0.5f},{0.2f,0.8f,0.8f}}, {{-0.5f,-0.5f,-0.5f},{0.2f,0.8f,0.8f}},
    {{ 0.5f,-0.5f, 0.5f},{0.2f,0.8f,0.8f}}, {{-0.5f,-0.5f, 0.5f},{0.2f,0.8f,0.8f}},
};

static const std::vector<Vertex> OUTLINE_VERTICES = [] {
    std::vector<Vertex> verts = CUBE_VERTICES;
    for (Vertex& v : verts)
        v.color = glm::vec3(1.0f, 0.85f, 0.1f);
    return verts;
}();

class Ch39App {
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow*           window_              = nullptr;
    VkInstance            instance_            = VK_NULL_HANDLE;
    VkSurfaceKHR          surface_             = VK_NULL_HANDLE;
    VkPhysicalDevice      physicalDevice_      = VK_NULL_HANDLE;
    VkDevice              device_              = VK_NULL_HANDLE;
    VkQueue               graphicsQueue_       = VK_NULL_HANDLE;
    VkQueue               presentQueue_        = VK_NULL_HANDLE;
    VkSwapchainKHR        swapchain_           = VK_NULL_HANDLE;
    VkRenderPass          renderPass_          = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout_      = VK_NULL_HANDLE;
    VkPipeline            fillPipeline_        = VK_NULL_HANDLE;
    VkPipeline            stencilWritePipeline_= VK_NULL_HANDLE;
    VkPipeline            outlinePipeline_     = VK_NULL_HANDLE;
    VkCommandPool         commandPool_         = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool_      = VK_NULL_HANDLE;

    VkImage        depthImage_       = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
    VkImageView    depthImageView_   = VK_NULL_HANDLE;
    VkFormat       depthFormat_      = VK_FORMAT_UNDEFINED;

    std::vector<VkDescriptorSet>  descriptorSets_;
    std::vector<VkDescriptorSet>  outlineDescriptorSets_;
    std::vector<VkBuffer>         uniformBuffers_;
    std::vector<VkDeviceMemory>   uniformBuffersMemory_;
    std::vector<void*>            uniformBuffersMapped_;
    std::vector<VkBuffer>         outlineUniformBuffers_;
    std::vector<VkDeviceMemory>   outlineUniformMemory_;
    std::vector<void*>            outlineUniformMapped_;
    std::vector<VkImage>          swapchainImages_;
    std::vector<VkImageView>      swapchainImageViews_;
    std::vector<VkFramebuffer>    framebuffers_;
    std::vector<VkCommandBuffer>  commandBuffers_;
    VkFormat                      swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D                    swapchainExtent_{};
    QueueFamilyIndices            queueIndices_;

    VkBuffer       cubeVertexBuffer_       = VK_NULL_HANDLE;
    VkDeviceMemory cubeVertexMemory_       = VK_NULL_HANDLE;
    VkBuffer       outlineVertexBuffer_    = VK_NULL_HANDLE;
    VkDeviceMemory outlineVertexMemory_    = VK_NULL_HANDLE;

    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence>     inFlightFences_;
    uint32_t                 currentFrame_ = 0;
    bool                     resized_      = false;
    InteractiveChapterTools  interactive_;

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT,
            "Ch39 - 模板缓冲描边（Stencil Outline）", nullptr, nullptr);
        interactive_.attachInput(window_);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch39App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan()
    {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physicalDevice_);
        createLogicalDevice(physicalDevice_, surface_, device_, graphicsQueue_,
                            presentQueue_, queueIndices_);
        createSwapchain();
        createImageViews();
        depthFormat_ = findDepthStencilFormat(physicalDevice_);
        if (depthFormat_ == VK_FORMAT_UNDEFINED)
            throw std::runtime_error("找不到深度/模板格式");
        if (!hasStencilComponent(depthFormat_))
            logFeatureFallback("模板分量", "D32_SFLOAT（无模板，描边将不可用）");
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipelines();
        createDepthStencilResources();
        createFramebuffers();
        createCommandPool();
        createVertexBuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
        InteractiveInitInfo ii{};
        ii.window = window_;
        ii.instance = instance_;
        ii.physicalDevice = physicalDevice_;
        ii.device = device_;
        ii.graphicsQueue = graphicsQueue_;
        ii.queueFamily = queueIndices_.graphicsFamily.value();
        ii.renderPass = renderPass_;
        ii.swapchainFormat = swapchainImageFormat_;
        ii.imageCount = static_cast<uint32_t>(swapchainImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setTarget(glm::vec3(0.0f));
        interactive_.camera().setDistance(4.0f);
        std::cout << "\n✅ 模板缓冲初始化完成（格式=" << depthFormat_ << "）\n";
    }

    void createDepthStencilResources()
    {
        createImage(physicalDevice_, device_, swapchainExtent_.width, swapchainExtent_.height,
                    depthFormat_, VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    depthImage_, depthImageMemory_);
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image    = depthImage_;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format   = depthFormat_;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (hasStencilComponent(depthFormat_))
            viewInfo.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(device_, &viewInfo, nullptr, &depthImageView_));
    }

    void destroyDepthStencilResources()
    {
        if (depthImageView_ != VK_NULL_HANDLE)
            vkDestroyImageView(device_, depthImageView_, nullptr);
        if (depthImage_ != VK_NULL_HANDLE)
            vkDestroyImage(device_, depthImage_, nullptr);
        if (depthImageMemory_ != VK_NULL_HANDLE)
            vkFreeMemory(device_, depthImageMemory_, nullptr);
        depthImageView_   = VK_NULL_HANDLE;
        depthImage_       = VK_NULL_HANDLE;
        depthImageMemory_ = VK_NULL_HANDLE;
    }

    void createSwapchain()
    {
        const SwapChainSupportDetails sc = querySwapChainSupport(physicalDevice_, surface_);
        const VkSurfaceFormatKHR fmt     = chooseSwapSurfaceFormat(sc.formats);
        const VkPresentModeKHR mode      = chooseSwapPresentMode(sc.presentModes);
        swapchainExtent_                 = chooseSwapExtent(sc.capabilities, window_);
        uint32_t imageCount              = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            imageCount = std::min(imageCount, sc.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{};
        ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface          = surface_;
        ci.minImageCount    = imageCount;
        ci.imageFormat      = fmt.format;
        ci.imageColorSpace  = fmt.colorSpace;
        ci.imageExtent      = swapchainExtent_;
        ci.imageArrayLayers = 1;
        ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.preTransform     = sc.capabilities.currentTransform;
        ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode      = mode;
        ci.clipped          = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
        swapchainImages_.resize(imageCount);
        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchainImages_.data());
        swapchainImageFormat_ = fmt.format;
    }

    void createImageViews()
    {
        swapchainImageViews_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image    = swapchainImages_[i];
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format   = swapchainImageFormat_;
            ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]));
        }
    }

    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format         = swapchainImageFormat_;
        colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format         = depthFormat_;
        depthAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 1;
        subpass.pColorAttachments       = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        VkSubpassDependency dep{};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass    = 0;
        dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
        VkRenderPassCreateInfo rpi{};
        rpi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = static_cast<uint32_t>(attachments.size());
        rpi.pAttachments    = attachments.data();
        rpi.subpassCount    = 1;
        rpi.pSubpasses      = &subpass;
        rpi.dependencyCount = 1;
        rpi.pDependencies   = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_));
    }

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding uboBinding{};
        uboBinding.binding         = 0;
        uboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.descriptorCount = 1;
        uboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings    = &uboBinding;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descriptorSetLayout_));
    }

    VkStencilOpState makeStencilReplace()
    {
        VkStencilOpState s{};
        s.failOp      = VK_STENCIL_OP_KEEP;
        s.passOp      = VK_STENCIL_OP_REPLACE;
        s.depthFailOp = VK_STENCIL_OP_KEEP;
        s.compareOp   = VK_COMPARE_OP_ALWAYS;
        s.compareMask = 0xFF;
        s.writeMask   = 0xFF;
        s.reference   = STENCIL_REF;
        return s;
    }

    VkStencilOpState makeStencilOutlineTest()
    {
        VkStencilOpState s{};
        s.failOp      = VK_STENCIL_OP_KEEP;
        s.passOp      = VK_STENCIL_OP_KEEP;
        s.depthFailOp = VK_STENCIL_OP_KEEP;
        s.compareOp   = VK_COMPARE_OP_NOT_EQUAL;
        s.compareMask = 0xFF;
        s.writeMask   = 0x00;
        s.reference   = STENCIL_REF;
        return s;
    }

    VkPipeline buildPipeline(VkPipelineColorBlendAttachmentState cba,
                             VkPipelineDepthStencilStateCreateInfo ds)
    {
        VkShaderModule vert = createShaderModuleFromFile(device_, "stencil.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "stencil.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};

        const auto bd = Vertex::getBindingDescription();
        const auto ad = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &bd;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(ad.size());
        vi.pVertexAttributeDescriptions    = ad.data();

        VkPipelineInputAssemblyStateCreateInfo ia{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_BACK_BIT;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments    = &cba;

        const std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynInfo{};
        dynInfo.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynInfo.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        dynInfo.pDynamicStates    = dyn.data();

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount          = 2;
        pi.pStages             = stages;
        pi.pVertexInputState   = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState      = &vs;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState   = &ms;
        pi.pDepthStencilState  = &ds;
        pi.pColorBlendState    = &cb;
        pi.pDynamicState       = &dynInfo;
        pi.layout              = pipelineLayout_;
        pi.renderPass          = renderPass_;
        VkPipeline pipeline    = VK_NULL_HANDLE;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        return pipeline;
    }

    void createGraphicsPipelines()
    {
        VkPipelineLayoutCreateInfo pli{};
        pli.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts    = &descriptorSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &pipelineLayout_));

        VkPipelineColorBlendAttachmentState fillBlend{};
        fillBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT;
        fillBlend.blendEnable    = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo fillDs{};
        fillDs.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        fillDs.depthTestEnable  = VK_TRUE;
        fillDs.depthWriteEnable = VK_TRUE;
        fillDs.depthCompareOp   = VK_COMPARE_OP_LESS;
        fillDs.stencilTestEnable = VK_FALSE;
        fillPipeline_ = buildPipeline(fillBlend, fillDs);

        VkPipelineColorBlendAttachmentState noColor{};
        noColor.colorWriteMask = 0;
        noColor.blendEnable    = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo stencilDs{};
        stencilDs.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        stencilDs.depthTestEnable  = VK_TRUE;
        stencilDs.depthWriteEnable = VK_TRUE;
        stencilDs.depthCompareOp   = VK_COMPARE_OP_LESS;
        stencilDs.stencilTestEnable = VK_TRUE;
        stencilDs.front             = makeStencilReplace();
        stencilDs.back              = stencilDs.front;
        stencilWritePipeline_       = buildPipeline(noColor, stencilDs);

        VkPipelineColorBlendAttachmentState outlineBlend = fillBlend;
        VkPipelineDepthStencilStateCreateInfo outlineDs{};
        outlineDs.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        outlineDs.depthTestEnable  = VK_FALSE;
        outlineDs.depthWriteEnable = VK_FALSE;
        outlineDs.stencilTestEnable = VK_TRUE;
        outlineDs.front             = makeStencilOutlineTest();
        outlineDs.back              = outlineDs.front;
        outlinePipeline_            = buildPipeline(outlineBlend, outlineDs);
    }

    void createFramebuffers()
    {
        framebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            std::array<VkImageView, 2> attachments = {swapchainImageViews_[i], depthImageView_};
            VkFramebufferCreateInfo ci{};
            ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass      = renderPass_;
            ci.attachmentCount = static_cast<uint32_t>(attachments.size());
            ci.pAttachments    = attachments.data();
            ci.width           = swapchainExtent_.width;
            ci.height          = swapchainExtent_.height;
            ci.layers          = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
        }
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{};
        ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
    }

    void uploadVertices(const std::vector<Vertex>& data, VkBuffer& buffer, VkDeviceMemory& memory)
    {
        const VkDeviceSize size = sizeof(data[0]) * data.size();
        VkBuffer staging        = VK_NULL_HANDLE;
        VkDeviceMemory stagingMem = VK_NULL_HANDLE;
        createBuffer(physicalDevice_, device_, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMem);
        void* mapped = nullptr;
        vkMapMemory(device_, stagingMem, 0, size, 0, &mapped);
        std::memcpy(mapped, data.data(), static_cast<size_t>(size));
        vkUnmapMemory(device_, stagingMem);
        createBuffer(physicalDevice_, device_, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);
        copyBuffer(device_, commandPool_, graphicsQueue_, staging, buffer, size);
        vkDestroyBuffer(device_, staging, nullptr);
        vkFreeMemory(device_, stagingMem, nullptr);
    }

    void createVertexBuffers()
    {
        uploadVertices(CUBE_VERTICES, cubeVertexBuffer_, cubeVertexMemory_);
        uploadVertices(OUTLINE_VERTICES, outlineVertexBuffer_, outlineVertexMemory_);
    }

    void createUniformBuffers()
    {
        const VkDeviceSize size = sizeof(UniformBufferObject);
        uniformBuffers_.resize(MAX_FRAMES);
        uniformBuffersMemory_.resize(MAX_FRAMES);
        uniformBuffersMapped_.resize(MAX_FRAMES);
        outlineUniformBuffers_.resize(MAX_FRAMES);
        outlineUniformMemory_.resize(MAX_FRAMES);
        outlineUniformMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physicalDevice_, device_, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers_[i], uniformBuffersMemory_[i]);
            vkMapMemory(device_, uniformBuffersMemory_[i], 0, size, 0, &uniformBuffersMapped_[i]);
            createBuffer(physicalDevice_, device_, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         outlineUniformBuffers_[i], outlineUniformMemory_[i]);
            vkMapMemory(device_, outlineUniformMemory_[i], 0, size, 0, &outlineUniformMapped_[i]);
        }
    }

    void createDescriptorPool()
    {
        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                      static_cast<uint32_t>(MAX_FRAMES * 2)};
        VkDescriptorPoolCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = 1;
        ci.pPoolSizes    = &poolSize;
        ci.maxSets       = static_cast<uint32_t>(MAX_FRAMES * 2);
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &descriptorPool_));
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES * 2, descriptorSetLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descriptorPool_;
        ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES * 2);
        ai.pSetLayouts        = layouts.data();
        std::vector<VkDescriptorSet> allSets(MAX_FRAMES * 2);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, allSets.data()));
        descriptorSets_.assign(allSets.begin(), allSets.begin() + MAX_FRAMES);
        outlineDescriptorSets_.assign(allSets.begin() + MAX_FRAMES, allSets.end());
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = uniformBuffers_[i];
            bi.offset = 0;
            bi.range  = sizeof(UniformBufferObject);
            VkWriteDescriptorSet write{};
            write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet          = descriptorSets_[i];
            write.dstBinding      = 0;
            write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo     = &bi;
            vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
            bi.buffer             = outlineUniformBuffers_[i];
            write.dstSet          = outlineDescriptorSets_[i];
            vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        }
    }

    void createCommandBuffers()
    {
        commandBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = commandPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()));
    }

    void createSyncObjects()
    {
        imageAvailableSems_.resize(MAX_FRAMES);
        renderFinishedSems_.resize(MAX_FRAMES);
        inFlightFences_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo semCI{};
        semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &semCI, nullptr, &imageAvailableSems_[i]));
            VK_CHECK(vkCreateSemaphore(device_, &semCI, nullptr, &renderFinishedSems_[i]));
            VK_CHECK(vkCreateFence(device_, &fenceCI, nullptr, &inFlightFences_[i]));
        }
    }

    UniformBufferObject makeUbo(float scale)
    {
        static auto start = std::chrono::high_resolution_clock::now();
        const float t = std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now() - start).count();
        UniformBufferObject ubo{};
        ubo.model      = glm::scale(glm::rotate(glm::mat4(1.0f), t * glm::radians(45.0f),
            glm::vec3(0, 1, 0)), glm::vec3(scale));
        const float aspect = static_cast<float>(swapchainExtent_.width) /
                             static_cast<float>(swapchainExtent_.height);
        ubo.view       = interactive_.camera().viewMatrix();
        ubo.projection = interactive_.camera().projectionMatrix(aspect, 45.0f, 0.1f, 10.0f);
        return ubo;
    }

    void updateUniformBuffers(uint32_t frame)
    {
        const UniformBufferObject fillUbo  = makeUbo(1.0f);
        const UniformBufferObject outlineUbo = makeUbo(1.08f);
        std::memcpy(uniformBuffersMapped_[frame], &fillUbo, sizeof(fillUbo));
        std::memcpy(outlineUniformMapped_[frame], &outlineUbo, sizeof(outlineUbo));
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, currentFrame_);

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color        = {{0.02f, 0.02f, 0.05f, 1.0f}};
        clearValues[1].depthStencil   = {1.0f, 0};

        VkRenderPassBeginInfo rpBI{};
        rpBI.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBI.renderPass      = renderPass_;
        rpBI.framebuffer     = framebuffers_[imageIndex];
        rpBI.renderArea      = {{0, 0}, swapchainExtent_};
        rpBI.clearValueCount = static_cast<uint32_t>(clearValues.size());
        rpBI.pClearValues    = clearValues.data();
        vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);

        const VkViewport vp{0.0f, 0.0f, static_cast<float>(swapchainExtent_.width),
            static_cast<float>(swapchainExtent_.height), 0.0f, 1.0f};
        const VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        VkDeviceSize offset = 0;
        const uint32_t vertexCount = static_cast<uint32_t>(CUBE_VERTICES.size());

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, stencilWritePipeline_);
        VkBuffer cubeBuf[] = {cubeVertexBuffer_};
        vkCmdBindVertexBuffers(cmd, 0, 1, cubeBuf, &offset);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_,
                                0, 1, &descriptorSets_[currentFrame_], 0, nullptr);
        vkCmdDraw(cmd, vertexCount, 1, 0, 0);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, outlinePipeline_);
        VkBuffer outlineBuf[] = {outlineVertexBuffer_};
        vkCmdBindVertexBuffers(cmd, 0, 1, outlineBuf, &offset);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_,
                                0, 1, &outlineDescriptorSets_[currentFrame_], 0, nullptr);
        vkCmdDraw(cmd, vertexCount, 1, 0, 0);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fillPipeline_);
        vkCmdBindVertexBuffers(cmd, 0, 1, cubeBuf, &offset);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_,
                                0, 1, &descriptorSets_[currentFrame_], 0, nullptr);
        vkCmdDraw(cmd, vertexCount, 1, 0, 0);

        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);
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
        updateUniformBuffers(currentFrame_);
        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordCommandBuffer(commandBuffers_[currentFrame_], imageIndex);

        VkSemaphore waitSems[]   = {imageAvailableSems_[currentFrame_]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore signalSems[] = {renderFinishedSems_[currentFrame_]};
        VkSubmitInfo submit{};
        submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount   = 1;
        submit.pWaitSemaphores      = waitSems;
        submit.pWaitDstStageMask    = waitStages;
        submit.commandBufferCount   = 1;
        submit.pCommandBuffers      = &commandBuffers_[currentFrame_];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores    = signalSems;
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &submit, inFlightFences_[currentFrame_]));

        VkSwapchainKHR swapchains[] = {swapchain_};
        VkPresentInfoKHR present{};
        present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores    = signalSems;
        present.swapchainCount     = 1;
        present.pSwapchains        = swapchains;
        present.pImageIndices      = &imageIndex;
        result = vkQueuePresentKHR(presentQueue_, &present);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        }
        interactive_.endFrame(currentFrame_);
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
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
        for (auto fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        destroyDepthStencilResources();
        for (auto iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        createDepthStencilResources();
        createFramebuffers();
        interactive_.onSwapchainRecreated(renderPass_, swapchainImageFormat_,
            static_cast<uint32_t>(swapchainImages_.size()));
    }

    void mainLoop()
    {
        std::cout << "🎨 旋转立方体 + 黄色模板描边（ESC 退出）\n";
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
        destroyDepthStencilResources();
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, uniformBuffers_[i], nullptr);
            vkFreeMemory(device_, uniformBuffersMemory_[i], nullptr);
            vkDestroyBuffer(device_, outlineUniformBuffers_[i], nullptr);
            vkFreeMemory(device_, outlineUniformMemory_[i], nullptr);
        }
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        vkDestroyBuffer(device_, cubeVertexBuffer_, nullptr);
        vkFreeMemory(device_, cubeVertexMemory_, nullptr);
        vkDestroyBuffer(device_, outlineVertexBuffer_, nullptr);
        vkFreeMemory(device_, outlineVertexMemory_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (auto fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyPipeline(device_, fillPipeline_, nullptr);
        vkDestroyPipeline(device_, stencilWritePipeline_, nullptr);
        vkDestroyPipeline(device_, outlinePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        for (auto iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        interactive_.shutdown(device_);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
        std::cout << "✅ 清理完成。\n";
    }
};

int main()
{
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << " 第39章：模板缓冲描边\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    Ch39App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
