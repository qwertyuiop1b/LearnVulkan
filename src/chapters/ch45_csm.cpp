/**
 * @file ch45_csm.cpp
 * @brief 第45章：级联阴影贴图（Cascaded Shadow Maps）
 *
 * 将视锥体沿深度方向切分为 3 级联，每级使用独立 shadow map（2D 数组），
 * 近处高分辨率、远处低分辨率，缓解透视 aliasing。
 * 分割采用 Practical Split Scheme（对数 + 均匀混合）。
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
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;
constexpr uint32_t SHADOW_DIM = 1024;
constexpr uint32_t CASCADE_COUNT = 3;

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
};

static const std::vector<Vertex> SCENE = {
    {{-4, -0.5f, -4}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}},
    {{4, -0.5f, -4}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}},
    {{4, -0.5f, 4}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}},
    {{-4, -0.5f, -4}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}},
    {{4, -0.5f, 4}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}},
    {{-4, -0.5f, 4}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}},
    {{-0.25f, -0.5f, -0.25f}, {0, 0, 1}, {0.9f, 0.25f, 0.25f}},
    {{0.25f, -0.5f, -0.25f}, {0, 0, 1}, {0.9f, 0.25f, 0.25f}},
    {{0.25f, 1.6f, -0.25f}, {0, 0, 1}, {0.9f, 0.25f, 0.25f}},
    {{-0.25f, -0.5f, -0.25f}, {0, 0, 1}, {0.9f, 0.25f, 0.25f}},
    {{0.25f, 1.6f, -0.25f}, {0, 0, 1}, {0.9f, 0.25f, 0.25f}},
    {{-0.25f, 1.6f, -0.25f}, {0, 0, 1}, {0.9f, 0.25f, 0.25f}},
    {{-1.6f, -0.5f, 0.4f}, {1, 0, 0}, {0.25f, 0.85f, 0.3f}},
    {{-1.2f, -0.5f, 0.4f}, {1, 0, 0}, {0.25f, 0.85f, 0.3f}},
    {{-1.2f, 2.1f, 0.4f}, {1, 0, 0}, {0.25f, 0.85f, 0.3f}},
    {{-1.6f, -0.5f, 0.4f}, {1, 0, 0}, {0.25f, 0.85f, 0.3f}},
    {{-1.2f, 2.1f, 0.4f}, {1, 0, 0}, {0.25f, 0.85f, 0.3f}},
    {{-1.6f, 2.1f, 0.4f}, {1, 0, 0}, {0.25f, 0.85f, 0.3f}},
    {{1.2f, -0.5f, -0.6f}, {0, 0, 1}, {0.3f, 0.35f, 0.95f}},
    {{1.6f, -0.5f, -0.6f}, {0, 0, 1}, {0.3f, 0.35f, 0.95f}},
    {{1.6f, 1.3f, -0.6f}, {0, 0, 1}, {0.3f, 0.35f, 0.95f}},
    {{1.2f, -0.5f, -0.6f}, {0, 0, 1}, {0.3f, 0.35f, 0.95f}},
    {{1.6f, 1.3f, -0.6f}, {0, 0, 1}, {0.3f, 0.35f, 0.95f}},
    {{1.2f, 1.3f, -0.6f}, {0, 0, 1}, {0.3f, 0.35f, 0.95f}},
};

struct SceneUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::mat4 lightView;
    alignas(16) glm::vec4 cascadeSplits;
    alignas(16) glm::mat4 lightSpaceMatrix[CASCADE_COUNT];
    alignas(16) glm::vec4 lightDir;
};

struct ShadowPC {
    glm::mat4 lightSpaceMatrix;
};

class Ch45App {
  public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

  private:
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    QueueFamilyIndices queueIndices_;
    VkExtent2D extent_{};
    VkFormat swapFormat_ = VK_FORMAT_UNDEFINED;
    DepthResources depth_{};
    glm::vec3 lightDir_ = glm::normalize(glm::vec3(-0.8f, -2.0f, -1.0f));

    VkImage shadowImage_ = VK_NULL_HANDLE;
    VkDeviceMemory shadowMemory_ = VK_NULL_HANDLE;
    VkImageView shadowArrayView_ = VK_NULL_HANDLE;
    std::array<VkImageView, CASCADE_COUNT> shadowLayerViews_{};
    VkSampler shadowSampler_ = VK_NULL_HANDLE;
    VkRenderPass shadowRenderPass_ = VK_NULL_HANDLE;
    std::array<VkFramebuffer, CASCADE_COUNT> shadowFramebuffers_{};
    VkPipelineLayout shadowPipeLayout_ = VK_NULL_HANDLE;
    VkPipeline shadowPipeline_ = VK_NULL_HANDLE;

    VkRenderPass sceneRenderPass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sceneSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout scenePipeLayout_ = VK_NULL_HANDLE;
    VkPipeline scenePipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneDescSets_;
    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> sceneUBOs_;
    std::vector<VkDeviceMemory> sceneUBOMem_;
    std::vector<void*> sceneUBOMapped_;

    std::vector<VkImage> swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<VkFramebuffer> sceneFramebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<VkSemaphore> imageAvailable_;
    std::vector<VkSemaphore> renderFinished_;
    std::vector<VkFence> inFlight_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;
    InteractiveChapterTools interactive_;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch45 - 级联阴影贴图（3 级联）", nullptr, nullptr);
        interactive_.attachInput(window_);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch45App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan() {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physicalDevice_);
        createLogicalDevice(physicalDevice_, surface_, device_, graphicsQueue_, presentQueue_, queueIndices_);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physicalDevice_, device_, extent_);
        createShadowResources();
        createShadowRenderPass();
        createShadowPipeline();
        createSceneRenderPass();
        createScenePipeline();
        createSceneFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createSceneUBOs();
        createDescriptorPool();
        createSceneDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
        InteractiveInitInfo ii{};
        ii.window = window_;
        ii.instance = instance_;
        ii.physicalDevice = physicalDevice_;
        ii.device = device_;
        ii.graphicsQueue = graphicsQueue_;
        ii.queueFamily = queueIndices_.graphicsFamily.value();
        ii.renderPass = sceneRenderPass_;
        ii.swapchainFormat = swapFormat_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setTarget(glm::vec3(0.0f));
        interactive_.camera().setDistance(7.0f);
        std::cout << "✅ CSM 初始化完成（" << CASCADE_COUNT << " 级联，" << SHADOW_DIM << "²）\n";
    }

    std::array<float, CASCADE_COUNT + 1> computeSplits(float nearZ, float farZ) const {
        std::array<float, CASCADE_COUNT + 1> splits{};
        const float lambda = 0.75f;
        for (uint32_t i = 0; i <= CASCADE_COUNT; ++i) {
            const float p = static_cast<float>(i) / static_cast<float>(CASCADE_COUNT);
            const float logSplit = nearZ * std::pow(farZ / nearZ, p);
            const float uniSplit = nearZ + (farZ - nearZ) * p;
            splits[i] = lambda * logSplit + (1.0f - lambda) * uniSplit;
        }
        return splits;
    }

    glm::mat4 getLightView() const {
        return glm::lookAt(-lightDir_ * 10.0f, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::mat4 fitOrthoToCascade(const glm::mat4& lightView,
                                float splitNear,
                                float splitFar,
                                const glm::mat4& invView,
                                const glm::mat4& proj) const {
        const glm::mat4 invViewProj = invView * glm::inverse(proj);
        std::array<glm::vec4, 8> corners{};
        int idx = 0;
        for (int x = 0; x < 2; ++x)
            for (int y = 0; y < 2; ++y)
                for (int z = 0; z < 2; ++z) {
                    glm::vec4 p = invViewProj * glm::vec4(x ? 1.0f : -1.0f, y ? 1.0f : -1.0f, z ? 1.0f : -1.0f, 1.0f);
                    corners[idx++] = p / p.w;
                }
        glm::vec3 minLS(1e9f), maxLS(-1e9f);
        for (const glm::vec4& c : corners) {
            const glm::vec4 ls = lightView * c;
            minLS = glm::min(minLS, glm::vec3(ls));
            maxLS = glm::max(maxLS, glm::vec3(ls));
        }
        const float margin = 2.0f;
        glm::mat4 lightProj = glm::ortho(
            minLS.x - margin, maxLS.x + margin, minLS.y - margin, maxLS.y + margin, -maxLS.z - 20.0f, -minLS.z + 20.0f);
        lightProj[1][1] *= -1.0f;
        (void)splitNear;
        (void)splitFar;
        return lightProj * lightView;
    }

    void updateUBO(uint32_t frame) {
        const float nearZ = 0.1f;
        const float farZ = 40.0f;
        const auto splits = computeSplits(nearZ, farZ);
        SceneUBO ubo{};
        ubo.model = glm::mat4(1.0f);
        const float aspect = extent_.width / static_cast<float>(extent_.height);
        ubo.view = interactive_.camera().viewMatrix();
        ubo.projection = interactive_.camera().projectionMatrix(aspect, 55.0f, nearZ, farZ);
        ubo.lightView = getLightView();
        ubo.cascadeSplits = glm::vec4(splits[1], splits[2], splits[3], nearZ);
        ubo.lightDir = glm::vec4(lightDir_, 0.0f);
        const glm::mat4 invView = glm::inverse(ubo.view);
        for (uint32_t i = 0; i < CASCADE_COUNT; ++i)
            ubo.lightSpaceMatrix[i] =
                fitOrthoToCascade(ubo.lightView, splits[i], splits[i + 1], invView, ubo.projection);
        std::memcpy(sceneUBOMapped_[frame], &ubo, sizeof(ubo));
    }

    void createShadowResources() {
        createImage(physicalDevice_,
                    device_,
                    SHADOW_DIM,
                    SHADOW_DIM,
                    VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    shadowImage_,
                    shadowMemory_,
                    1,
                    CASCADE_COUNT,
                    VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT);
        VkImageViewCreateInfo arrayView{};
        arrayView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        arrayView.image = shadowImage_;
        arrayView.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        arrayView.format = VK_FORMAT_D32_SFLOAT;
        arrayView.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, CASCADE_COUNT};
        VK_CHECK(vkCreateImageView(device_, &arrayView, nullptr, &shadowArrayView_));
        for (uint32_t i = 0; i < CASCADE_COUNT; ++i) {
            VkImageViewCreateInfo layerView = arrayView;
            layerView.viewType = VK_IMAGE_VIEW_TYPE_2D;
            layerView.subresourceRange.baseArrayLayer = i;
            layerView.subresourceRange.layerCount = 1;
            VK_CHECK(vkCreateImageView(device_, &layerView, nullptr, &shadowLayerViews_[i]));
        }
        VkSamplerCreateInfo samp{};
        samp.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samp.magFilter = VK_FILTER_LINEAR;
        samp.minFilter = VK_FILTER_LINEAR;
        samp.addressModeU = samp.addressModeV = samp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samp.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK(vkCreateSampler(device_, &samp, nullptr, &shadowSampler_));
    }

    void createShadowRenderPass() {
        VkAttachmentDescription depth{};
        depth.format = VK_FORMAT_D32_SFLOAT;
        depth.samples = VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentReference depthRef{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.pDepthStencilAttachment = &depthRef;
        std::array<VkSubpassDependency, 2> deps{};
        deps[0] = {VK_SUBPASS_EXTERNAL,
                   0,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                   VK_ACCESS_SHADER_READ_BIT,
                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_DEPENDENCY_BY_REGION_BIT};
        deps[1] = {0,
                   VK_SUBPASS_EXTERNAL,
                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT,
                   VK_DEPENDENCY_BY_REGION_BIT};
        VkRenderPassCreateInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp.attachmentCount = 1;
        rp.pAttachments = &depth;
        rp.subpassCount = 1;
        rp.pSubpasses = &subpass;
        rp.dependencyCount = static_cast<uint32_t>(deps.size());
        rp.pDependencies = deps.data();
        VK_CHECK(vkCreateRenderPass(device_, &rp, nullptr, &shadowRenderPass_));
        for (uint32_t i = 0; i < CASCADE_COUNT; ++i) {
            VkFramebufferCreateInfo fb{};
            fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb.renderPass = shadowRenderPass_;
            fb.attachmentCount = 1;
            fb.pAttachments = &shadowLayerViews_[i];
            fb.width = SHADOW_DIM;
            fb.height = SHADOW_DIM;
            fb.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &fb, nullptr, &shadowFramebuffers_[i]));
        }
    }

    VkShaderModule loadShader(const char* name) {
        return createShaderModuleFromFile(device_, name);
    }

    void createShadowPipeline() {
        VkShaderModule vert = loadShader("csm_shadow.vert.spv");
        VkShaderModule frag = loadShader("csm_shadow.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                      nullptr,
                                                      0,
                                                      VK_SHADER_STAGE_VERTEX_BIT,
                                                      vert,
                                                      "main",
                                                      nullptr},
                                                     {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                      nullptr,
                                                      0,
                                                      VK_SHADER_STAGE_FRAGMENT_BIT,
                                                      frag,
                                                      "main",
                                                      nullptr}};
        VkVertexInputBindingDescription bind{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = 1;
        vi.pVertexAttributeDescriptions = &attr;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                  nullptr,
                                                  0,
                                                  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                  VK_FALSE};
        VkViewport vp{0, 0, (float)SHADOW_DIM, (float)SHADOW_DIM, 0, 1};
        VkRect2D sc{{0, 0}, {SHADOW_DIM, SHADOW_DIM}};
        VkPipelineViewportStateCreateInfo vps{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &vp, 1, &sc};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        rs.depthBiasEnable = VK_TRUE;
        rs.depthBiasConstantFactor = 1.5f;
        rs.depthBiasSlopeFactor = 2.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                nullptr,
                                                0,
                                                VK_SAMPLE_COUNT_1_BIT,
                                                VK_FALSE,
                                                1.0f,
                                                nullptr,
                                                VK_FALSE,
                                                VK_FALSE};
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 0;
        VkPushConstantRange pc{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPC)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device_, &pl, nullptr, &shadowPipeLayout_));
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.stageCount = 2;
        gp.pStages = stages;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vps;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cb;
        gp.layout = shadowPipeLayout_;
        gp.renderPass = shadowRenderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &shadowPipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }

    void createSceneRenderPass() {
        VkAttachmentDescription color{};
        color.format = swapFormat_;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentDescription depthAtt{};
        depthAtt.format = depth_.format;
        depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference dref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &cref;
        sub.pDepthStencilAttachment = &dref;
        VkSubpassDependency dep{
            VK_SUBPASS_EXTERNAL,
            0,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
        std::array<VkAttachmentDescription, 2> atts = {color, depthAtt};
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.attachmentCount = static_cast<uint32_t>(atts.size());
        rp.pAttachments = atts.data();
        rp.subpassCount = 1;
        rp.pSubpasses = &sub;
        rp.dependencyCount = 1;
        rp.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rp, nullptr, &sceneRenderPass_));
    }

    void createScenePipeline() {
        VkShaderModule vert = loadShader("csm_scene.vert.spv");
        VkShaderModule frag = loadShader("csm_scene.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                      nullptr,
                                                      0,
                                                      VK_SHADER_STAGE_VERTEX_BIT,
                                                      vert,
                                                      "main",
                                                      nullptr},
                                                     {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                      nullptr,
                                                      0,
                                                      VK_SHADER_STAGE_FRAGMENT_BIT,
                                                      frag,
                                                      "main",
                                                      nullptr}};
        VkVertexInputBindingDescription bind{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 3> attrs{};
        attrs[0].location = 0;
        attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(Vertex, pos);
        attrs[1].location = 1;
        attrs[1].binding = 0;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = offsetof(Vertex, normal);
        attrs[2].location = 2;
        attrs[2].binding = 0;
        attrs[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[2].offset = offsetof(Vertex, color);
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
        vi.pVertexAttributeDescriptions = attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                  nullptr,
                                                  0,
                                                  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                  VK_FALSE};
        VkPipelineViewportStateCreateInfo vps{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                nullptr,
                                                0,
                                                VK_SAMPLE_COUNT_1_BIT,
                                                VK_FALSE,
                                                1.0f,
                                                nullptr,
                                                VK_FALSE,
                                                VK_FALSE};
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dynStates};
        std::array<VkDescriptorSetLayoutBinding, 2> binds{};
        binds[0].binding = 0;
        binds[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binds[0].descriptorCount = 1;
        binds[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        binds[1].binding = 1;
        binds[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binds[1].descriptorCount = 1;
        binds[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsl.bindingCount = static_cast<uint32_t>(binds.size());
        dsl.pBindings = binds.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dsl, nullptr, &sceneSetLayout_));
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &sceneSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pl, nullptr, &scenePipeLayout_));
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.stageCount = 2;
        gp.pStages = stages;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vps;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cb;
        gp.pDynamicState = &dyn;
        gp.layout = scenePipeLayout_;
        gp.renderPass = sceneRenderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &scenePipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginInfo);
        interactive_.beginGpuSection(cmd, currentFrame_);
        SceneUBO ubo{};
        std::memcpy(&ubo, sceneUBOMapped_[currentFrame_], sizeof(ubo));
        VkClearValue clearDepth{};
        clearDepth.depthStencil = {1.0f, 0};
        VkBuffer vertexBuffers[] = {vertexBuffer_};
        VkDeviceSize vertexOffsets[] = {0};
        for (uint32_t c = 0; c < CASCADE_COUNT; ++c) {
            VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rp.renderPass = shadowRenderPass_;
            rp.framebuffer = shadowFramebuffers_[c];
            rp.renderArea = {{0, 0}, {SHADOW_DIM, SHADOW_DIM}};
            rp.clearValueCount = 1;
            rp.pClearValues = &clearDepth;
            vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_);
            ShadowPC pc{ubo.lightSpaceMatrix[c]};
            vkCmdPushConstants(cmd, shadowPipeLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, vertexOffsets);
            vkCmdDraw(cmd, static_cast<uint32_t>(SCENE.size()), 1, 0, 0);
            vkCmdEndRenderPass(cmd);
        }
        std::array<VkClearValue, 2> clears{};
        clears[0].color = {{0.35f, 0.55f, 0.85f, 1.0f}};
        clears[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo sceneRp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        sceneRp.renderPass = sceneRenderPass_;
        sceneRp.framebuffer = sceneFramebuffers_[imageIndex];
        sceneRp.renderArea = {{0, 0}, extent_};
        sceneRp.clearValueCount = static_cast<uint32_t>(clears.size());
        sceneRp.pClearValues = clears.data();
        vkCmdBeginRenderPass(cmd, &sceneRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeline_);
        VkViewport vp{0, 0, (float)extent_.width, (float)extent_.height, 0, 1};
        VkRect2D sc{{0, 0}, extent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeLayout_, 0, 1, &sceneDescSets_[currentFrame_], 0, nullptr);
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, vertexOffsets);
        vkCmdDraw(cmd, static_cast<uint32_t>(SCENE.size()), 1, 0, 0);
        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);
        interactive_.endGpuSection(cmd, currentFrame_);
        vkEndCommandBuffer(cmd);
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlight_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t img = 0;
        VkResult acq = vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX, imageAvailable_[currentFrame_], VK_NULL_HANDLE, &img);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }
        updateUBO(currentFrame_);
        vkResetFences(device_, 1, &inFlight_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordCommandBuffer(commandBuffers_[currentFrame_], img);
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &imageAvailable_[currentFrame_];
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffers_[currentFrame_];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished_[currentFrame_];
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &submit, inFlight_[currentFrame_]));
        VkSwapchainKHR sc[] = {swapchain_};
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &renderFinished_[currentFrame_];
        present.swapchainCount = 1;
        present.pSwapchains = sc;
        present.pImageIndices = &img;
        VkResult pr = vkQueuePresentKHR(presentQueue_, &present);
        if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        }
        interactive_.endFrame(currentFrame_);
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void mainLoop() {
        std::cout << "🌑 CSM 运行中（ESC 退出）\n";
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

    void createSwapchain() {
        auto sc = querySwapChainSupport(physicalDevice_, surface_);
        auto fmt = chooseSwapSurfaceFormat(sc.formats);
        auto mode = chooseSwapPresentMode(sc.presentModes);
        extent_ = chooseSwapExtent(sc.capabilities, window_);
        uint32_t count = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            count = std::min(count, sc.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = surface_;
        ci.minImageCount = count;
        ci.imageFormat = fmt.format;
        ci.imageColorSpace = fmt.colorSpace;
        ci.imageExtent = extent_;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = sc.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = mode;
        ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        vkGetSwapchainImagesKHR(device_, swapchain_, &count, nullptr);
        swapImages_.resize(count);
        vkGetSwapchainImagesKHR(device_, swapchain_, &count, swapImages_.data());
        swapFormat_ = fmt.format;
    }

    void createImageViews() {
        swapViews_.resize(swapImages_.size());
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            ci.image = swapImages_[i];
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = swapFormat_;
            ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &swapViews_[i]));
        }
    }

    void createSceneFramebuffers() {
        sceneFramebuffers_.resize(swapViews_.size());
        for (size_t i = 0; i < swapViews_.size(); ++i) {
            std::array<VkImageView, 2> att = {swapViews_[i], depth_.view};
            VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fb.renderPass = sceneRenderPass_;
            fb.attachmentCount = static_cast<uint32_t>(att.size());
            fb.pAttachments = att.data();
            fb.width = extent_.width;
            fb.height = extent_.height;
            fb.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &fb, nullptr, &sceneFramebuffers_[i]));
        }
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
    }

    void createVertexBuffer() {
        const VkDeviceSize sz = sizeof(Vertex) * SCENE.size();
        createBuffer(physicalDevice_,
                     device_,
                     sz,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_,
                     vertexMemory_);
        void* mapped = nullptr;
        vkMapMemory(device_, vertexMemory_, 0, sz, 0, &mapped);
        std::memcpy(mapped, SCENE.data(), static_cast<size_t>(sz));
        vkUnmapMemory(device_, vertexMemory_);
    }

    void createSceneUBOs() {
        sceneUBOs_.resize(MAX_FRAMES);
        sceneUBOMem_.resize(MAX_FRAMES);
        sceneUBOMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physicalDevice_,
                         device_,
                         sizeof(SceneUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         sceneUBOs_[i],
                         sceneUBOMem_[i]);
            vkMapMemory(device_, sceneUBOMem_[i], 0, sizeof(SceneUBO), 0, &sceneUBOMapped_[i]);
        }
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> sizes{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[0].descriptorCount = MAX_FRAMES;
        sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = MAX_FRAMES;
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        ci.pPoolSizes = sizes.data();
        ci.maxSets = MAX_FRAMES;
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &descPool_));
    }

    void createSceneDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES, sceneSetLayout_);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descPool_;
        ai.descriptorSetCount = MAX_FRAMES;
        ai.pSetLayouts = layouts.data();
        sceneDescSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, sceneDescSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{sceneUBOs_[i], 0, sizeof(SceneUBO)};
            VkDescriptorImageInfo ii{shadowSampler_, shadowArrayView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::array<VkWriteDescriptorSet, 2> writes{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = sceneDescSets_[i];
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &bi;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = sceneDescSets_[i];
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].pImageInfo = &ii;
            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    void createCommandBuffers() {
        commandBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = MAX_FRAMES;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()));
    }

    void createSyncObjects() {
        imageAvailable_.resize(MAX_FRAMES);
        renderFinished_.resize(MAX_FRAMES);
        inFlight_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &imageAvailable_[i]));
            VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &renderFinished_[i]));
            VK_CHECK(vkCreateFence(device_, &fi, nullptr, &inFlight_[i]));
        }
    }

    void recreateSwapchain() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        for (auto fb : sceneFramebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        destroyDepthResources(device_, depth_);
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physicalDevice_, device_, extent_);
        createSceneFramebuffers();
        interactive_.onSwapchainRecreated(sceneRenderPass_, swapFormat_, static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup() {
        vkDestroySampler(device_, shadowSampler_, nullptr);
        vkDestroyImageView(device_, shadowArrayView_, nullptr);
        for (auto v : shadowLayerViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroyImage(device_, shadowImage_, nullptr);
        vkFreeMemory(device_, shadowMemory_, nullptr);
        for (auto fb : shadowFramebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyRenderPass(device_, shadowRenderPass_, nullptr);
        vkDestroyPipeline(device_, shadowPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, shadowPipeLayout_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, sceneUBOs_[i], nullptr);
            vkFreeMemory(device_, sceneUBOMem_[i], nullptr);
        }
        vkDestroyDescriptorPool(device_, descPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, sceneSetLayout_, nullptr);
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMemory_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailable_[i], nullptr);
            vkDestroySemaphore(device_, renderFinished_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (auto fb : sceneFramebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        destroyDepthResources(device_, depth_);
        vkDestroyPipeline(device_, scenePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, scenePipeLayout_, nullptr);
        vkDestroyRenderPass(device_, sceneRenderPass_, nullptr);
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        interactive_.shutdown(device_);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════\n";
    std::cout << " 第45章：级联阴影贴图（CSM，3 级联 Practical Split）\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    Ch45App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
    return 0;
}
