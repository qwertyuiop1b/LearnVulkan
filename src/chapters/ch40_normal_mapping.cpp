/**
 * @file ch40_normal_mapping.cpp
 * @brief 第40章：法线贴图（Normal Mapping）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【法线贴图原理】
 *
 *  法线贴图存储每个纹素的表面法线（切线空间），片段着色器将其变换到
 *  世界空间后与光照方向做点积，模拟凹凸细节而不增加几何复杂度。
 *
 * 【TBN 矩阵】
 *
 *    T = 切线 (Tangent)    → U 纹理方向
 *    B = 副切线 (Bitangent) → V 纹理方向
 *    N = 法线 (Normal)     → 表面法线
 *
 *    worldNormal = normalize(TBN * texNormal)
 *
 * 【描述符布局】
 *
 *  binding 0: UBO (model/view/projection)
 *  binding 1: diffuseMap  (brick_diffuse.png)
 *  binding 2: normalMap   (brick_normal.png)
 *  binding 3: Lighting UBO (lightDir, viewPos)
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan_tutorial/features.hpp>
#include <vulkan_tutorial/texture_loader.hpp>
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

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

struct LightingUBO {
    alignas(16) glm::vec3 lightDir;
    alignas(16) glm::vec3 viewPos;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 tangent;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription d{};
        d.binding   = 0;
        d.stride    = sizeof(Vertex);
        d.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return d;
    }

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 4> a{};
        a[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
        a[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
        a[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord)};
        a[3] = {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent)};
        return a;
    }
};

// XY 平面四边形，带切线基
static const std::vector<Vertex> PLANE_VERTICES = {
    {{-1.0f, -1.0f, 0.0f}, {0, 0, 1}, {0.0f, 0.0f}, {1, 0, 0}},
    {{ 1.0f, -1.0f, 0.0f}, {0, 0, 1}, {2.0f, 0.0f}, {1, 0, 0}},
    {{ 1.0f,  1.0f, 0.0f}, {0, 0, 1}, {2.0f, 2.0f}, {1, 0, 0}},
    {{-1.0f,  1.0f, 0.0f}, {0, 0, 1}, {0.0f, 2.0f}, {1, 0, 0}},
};

static const std::vector<uint16_t> PLANE_INDICES = {0, 1, 2, 2, 3, 0};

class Ch40App {
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
    VkPipeline            pipeline_            = VK_NULL_HANDLE;
    VkCommandPool         commandPool_         = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool_      = VK_NULL_HANDLE;

    DepthResources depth_{};
    TextureImage   diffuseTexture_{};
    TextureImage   normalTexture_{};

    std::vector<VkDescriptorSet>  descriptorSets_;
    std::vector<VkBuffer>         uniformBuffers_;
    std::vector<VkDeviceMemory>   uniformBuffersMemory_;
    std::vector<void*>            uniformBuffersMapped_;
    std::vector<VkBuffer>         lightingBuffers_;
    std::vector<VkDeviceMemory>   lightingBuffersMemory_;
    std::vector<void*>            lightingBuffersMapped_;
    std::vector<VkImage>          swapchainImages_;
    std::vector<VkImageView>      swapchainImageViews_;
    std::vector<VkFramebuffer>    framebuffers_;
    std::vector<VkCommandBuffer>  commandBuffers_;
    VkFormat                      swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D                    swapchainExtent_{};
    QueueFamilyIndices            queueIndices_;

    VkBuffer       vertexBuffer_       = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory_ = VK_NULL_HANDLE;
    VkBuffer       indexBuffer_        = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory_  = VK_NULL_HANDLE;

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
            "Ch40 - 法线贴图（砖墙平面）", nullptr, nullptr);
        interactive_.attachInput(window_);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch40App*>(glfwGetWindowUserPointer(w))->resized_ = true;
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
        depth_.format = findDepthFormat(physicalDevice_);
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        depth_ = createDepthResources(physicalDevice_, device_, swapchainExtent_);
        createFramebuffers();
        createCommandPool();
        loadTextures();  // 需要 commandPool_ 上传纹理
        createVertexBuffer();
        createIndexBuffer();
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
        std::cout << "\n✅ 法线贴图初始化完成\n";
    }

    void loadTextures()
    {
        const ImageData diffuseFallback = generateBrickDiffuse(256);
        const ImageData normalFallback  = generateBrickNormal(256);
        const ImageData diffuseData = loadImageWithFallback(
            "textures/brick_diffuse.png", diffuseFallback);
        const ImageData normalData = loadImageWithFallback(
            "textures/brick_normal.png", normalFallback);
        diffuseTexture_ = createTextureFromImageData(
            physicalDevice_, device_, commandPool_, graphicsQueue_, diffuseData, true);
        normalTexture_ = createTextureFromImageData(
            physicalDevice_, device_, commandPool_, graphicsQueue_, normalData, false);
        std::cout << "📦 漫反射 " << diffuseTexture_.width << "x" << diffuseTexture_.height
                  << "  法线 " << normalTexture_.width << "x" << normalTexture_.height << "\n";
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
        depthAttachment.format         = depth_.format;
        depthAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
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
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[2].binding         = 2;
        bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[3].binding         = 3;
        bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = static_cast<uint32_t>(bindings.size());
        ci.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descriptorSetLayout_));
    }

    void createGraphicsPipeline()
    {
        VkShaderModule vert = createShaderModuleFromFile(device_, "normal_map.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "normal_map.frag.spv");
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
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments    = &cba;

        const std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynInfo{};
        dynInfo.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynInfo.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        dynInfo.pDynamicStates    = dyn.data();

        VkPipelineLayoutCreateInfo pli{};
        pli.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts    = &descriptorSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &pipelineLayout_));

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
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }

    void createFramebuffers()
    {
        framebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            std::array<VkImageView, 2> attachments = {swapchainImageViews_[i], depth_.view};
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

    void createVertexBuffer()
    {
        const VkDeviceSize size = sizeof(PLANE_VERTICES[0]) * PLANE_VERTICES.size();
        VkBuffer staging        = VK_NULL_HANDLE;
        VkDeviceMemory stagingMem = VK_NULL_HANDLE;
        createBuffer(physicalDevice_, device_, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMem);
        void* mapped = nullptr;
        vkMapMemory(device_, stagingMem, 0, size, 0, &mapped);
        std::memcpy(mapped, PLANE_VERTICES.data(), static_cast<size_t>(size));
        vkUnmapMemory(device_, stagingMem);
        createBuffer(physicalDevice_, device_, size,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer_, vertexBufferMemory_);
        copyBuffer(device_, commandPool_, graphicsQueue_, staging, vertexBuffer_, size);
        vkDestroyBuffer(device_, staging, nullptr);
        vkFreeMemory(device_, stagingMem, nullptr);
    }

    void createIndexBuffer()
    {
        const VkDeviceSize size = sizeof(PLANE_INDICES[0]) * PLANE_INDICES.size();
        VkBuffer staging        = VK_NULL_HANDLE;
        VkDeviceMemory stagingMem = VK_NULL_HANDLE;
        createBuffer(physicalDevice_, device_, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMem);
        void* mapped = nullptr;
        vkMapMemory(device_, stagingMem, 0, size, 0, &mapped);
        std::memcpy(mapped, PLANE_INDICES.data(), static_cast<size_t>(size));
        vkUnmapMemory(device_, stagingMem);
        createBuffer(physicalDevice_, device_, size,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer_, indexBufferMemory_);
        copyBuffer(device_, commandPool_, graphicsQueue_, staging, indexBuffer_, size);
        vkDestroyBuffer(device_, staging, nullptr);
        vkFreeMemory(device_, stagingMem, nullptr);
    }

    void createUniformBuffers()
    {
        const VkDeviceSize uboSize      = sizeof(UniformBufferObject);
        const VkDeviceSize lightingSize  = sizeof(LightingUBO);
        uniformBuffers_.resize(MAX_FRAMES);
        uniformBuffersMemory_.resize(MAX_FRAMES);
        uniformBuffersMapped_.resize(MAX_FRAMES);
        lightingBuffers_.resize(MAX_FRAMES);
        lightingBuffersMemory_.resize(MAX_FRAMES);
        lightingBuffersMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physicalDevice_, device_, uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers_[i], uniformBuffersMemory_[i]);
            vkMapMemory(device_, uniformBuffersMemory_[i], 0, uboSize, 0, &uniformBuffersMapped_[i]);
            createBuffer(physicalDevice_, device_, lightingSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         lightingBuffers_[i], lightingBuffersMemory_[i]);
            vkMapMemory(device_, lightingBuffersMemory_[i], 0, lightingSize, 0,
                        &lightingBuffersMapped_[i]);
        }
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(MAX_FRAMES * 2)};
        poolSizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        static_cast<uint32_t>(MAX_FRAMES * 2)};
        VkDescriptorPoolCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        ci.pPoolSizes    = poolSizes.data();
        ci.maxSets       = static_cast<uint32_t>(MAX_FRAMES);
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &descriptorPool_));
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES, descriptorSetLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descriptorPool_;
        ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES);
        ai.pSetLayouts        = layouts.data();
        descriptorSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, descriptorSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo uboInfo{};
            uboInfo.buffer = uniformBuffers_[i];
            uboInfo.offset = 0;
            uboInfo.range  = sizeof(UniformBufferObject);
            VkDescriptorBufferInfo lightInfo{};
            lightInfo.buffer = lightingBuffers_[i];
            lightInfo.offset = 0;
            lightInfo.range  = sizeof(LightingUBO);
            VkDescriptorImageInfo diffuseInfo{};
            diffuseInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            diffuseInfo.imageView   = diffuseTexture_.view;
            diffuseInfo.sampler     = diffuseTexture_.sampler;
            VkDescriptorImageInfo normalInfo{};
            normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            normalInfo.imageView   = normalTexture_.view;
            normalInfo.sampler     = normalTexture_.sampler;
            std::array<VkWriteDescriptorSet, 4> writes{};
            writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                         descriptorSets_[i], 0, 0, 1,
                         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr};
            writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                         descriptorSets_[i], 1, 0, 1,
                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &diffuseInfo, nullptr, nullptr};
            writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                         descriptorSets_[i], 2, 0, 1,
                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &normalInfo, nullptr, nullptr};
            writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                         descriptorSets_[i], 3, 0, 1,
                         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &lightInfo, nullptr};
            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
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

    void updateUniformBuffers(uint32_t frame)
    {
        static auto start = std::chrono::high_resolution_clock::now();
        const float t = std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now() - start).count();
        UniformBufferObject ubo{};
        ubo.model      = glm::rotate(glm::mat4(1.0f), t * glm::radians(15.0f), glm::vec3(0, 0, 1));
        const float aspect = static_cast<float>(swapchainExtent_.width) /
                             static_cast<float>(swapchainExtent_.height);
        ubo.view       = interactive_.camera().viewMatrix();
        ubo.projection = interactive_.camera().projectionMatrix(aspect, 45.0f, 0.1f, 10.0f);
        std::memcpy(uniformBuffersMapped_[frame], &ubo, sizeof(ubo));
        LightingUBO lighting{};
        lighting.lightDir = glm::normalize(glm::vec3(-0.4f, -0.8f, -0.3f));
        lighting.viewPos  = interactive_.camera().eyePosition();
        std::memcpy(lightingBuffersMapped_[frame], &lighting, sizeof(lighting));
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

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        const VkViewport vp{0.0f, 0.0f, static_cast<float>(swapchainExtent_.width),
            static_cast<float>(swapchainExtent_.height), 0.0f, 1.0f};
        const VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_,
                                0, 1, &descriptorSets_[currentFrame_], 0, nullptr);
        VkDeviceSize offset = 0;
        VkBuffer vb[] = {vertexBuffer_};
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, &offset);
        vkCmdBindIndexBuffer(cmd, indexBuffer_, 0, VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(PLANE_INDICES.size()), 1, 0, 0, 0);
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
        destroyDepthResources(device_, depth_);
        for (auto iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physicalDevice_, device_, swapchainExtent_);
        createFramebuffers();
        interactive_.onSwapchainRecreated(renderPass_, swapchainImageFormat_,
            static_cast<uint32_t>(swapchainImages_.size()));
    }

    void mainLoop()
    {
        std::cout << "🎨 砖墙法线贴图平面（ESC 退出）\n";
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
        destroyTexture(device_, diffuseTexture_);
        destroyTexture(device_, normalTexture_);
        destroyDepthResources(device_, depth_);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, uniformBuffers_[i], nullptr);
            vkFreeMemory(device_, uniformBuffersMemory_[i], nullptr);
            vkDestroyBuffer(device_, lightingBuffers_[i], nullptr);
            vkFreeMemory(device_, lightingBuffersMemory_[i], nullptr);
        }
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        vkDestroyBuffer(device_, indexBuffer_, nullptr);
        vkFreeMemory(device_, indexBufferMemory_, nullptr);
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexBufferMemory_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (auto fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyPipeline(device_, pipeline_, nullptr);
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
    std::cout << " 第40章：法线贴图\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    Ch40App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
