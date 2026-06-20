/**
 * @file ch48_oit.cpp
 * @brief 第48章：加权混合顺序无关透明（Weighted Blended OIT）
 *
 * 三通道渲染：
 *   Pass 1 — 不透明背景写入 opaque 纹理
 *   Pass 2 — MRT 累积 accumulation + revealage（重叠透明四边形）
 *   Pass 3 — 全屏合成到交换链
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

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;

struct OitVertex {
    glm::vec3 pos;
    glm::vec3 color;
    float alpha;
};

struct SceneUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

static std::vector<OitVertex> buildQuad(glm::vec3 center, glm::vec2 size, glm::vec3 color, float alpha) {
    const float hx = size.x * 0.5f;
    const float hy = size.y * 0.5f;
    return {
        {{center.x - hx, center.y - hy, center.z}, color, alpha},
        {{center.x + hx, center.y - hy, center.z}, color, alpha},
        {{center.x + hx, center.y + hy, center.z}, color, alpha},
        {{center.x - hx, center.y - hy, center.z}, color, alpha},
        {{center.x + hx, center.y + hy, center.z}, color, alpha},
        {{center.x - hx, center.y + hy, center.z}, color, alpha},
    };
}

class Ch48App {
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
    QueueFamilyIndices queueIndices_{};
    VkFormat swapchainFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    DepthResources depth_{};

    VkImage opaqueImage_ = VK_NULL_HANDLE;
    VkDeviceMemory opaqueMemory_ = VK_NULL_HANDLE;
    VkImageView opaqueView_ = VK_NULL_HANDLE;
    VkImage accumImage_ = VK_NULL_HANDLE;
    VkDeviceMemory accumMemory_ = VK_NULL_HANDLE;
    VkImageView accumView_ = VK_NULL_HANDLE;
    VkImage revealImage_ = VK_NULL_HANDLE;
    VkDeviceMemory revealMemory_ = VK_NULL_HANDLE;
    VkImageView revealView_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;

    VkRenderPass opaqueRenderPass_ = VK_NULL_HANDLE;
    VkRenderPass oitRenderPass_ = VK_NULL_HANDLE;
    VkRenderPass compositeRenderPass_ = VK_NULL_HANDLE;
    VkFramebuffer opaqueFramebuffer_ = VK_NULL_HANDLE;
    VkFramebuffer oitFramebuffer_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout sceneSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout compositeSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout scenePipeLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout compositePipeLayout_ = VK_NULL_HANDLE;
    VkPipeline opaquePipeline_ = VK_NULL_HANDLE;
    VkPipeline oitPipeline_ = VK_NULL_HANDLE;
    VkPipeline compositePipeline_ = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneDescSets_;
    VkDescriptorSet compositeDescSet_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> sceneUBOs_;
    std::vector<VkDeviceMemory> sceneUBOMemories_;
    std::vector<void*> sceneUBOMapped_;

    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;
    uint32_t opaqueVertexCount_ = 0;
    uint32_t transparentVertexCount_ = 0;

    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkFramebuffer> compositeFramebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;
    InteractiveChapterTools interactive_;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch48 - 加权混合 OIT（MRT 累积 + 合成）", nullptr, nullptr);
        interactive_.attachInput(window_);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch48App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan() {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physicalDevice_);
        createLogicalDevice(physicalDevice_, surface_, device_, graphicsQueue_, presentQueue_, queueIndices_);
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
        ii.renderPass = compositeRenderPass_;
        ii.swapchainFormat = swapchainFormat_;
        ii.imageCount = static_cast<uint32_t>(swapchainImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setTarget(glm::vec3(0.0f));
        interactive_.camera().setDistance(5.0f);
        std::cout << "\n✅ OIT 初始化完成（accumulation + revealage MRT）\n";
    }

    void createSwapchain() {
        SwapChainSupportDetails support = querySwapChainSupport(physicalDevice_, surface_);
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(support.presentModes);
        swapchainExtent_ = chooseSwapExtent(support.capabilities, window_);
        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
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
        const uint32_t queueFamilies[] = {queueIndices_.graphicsFamily.value(), queueIndices_.presentFamily.value()};
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

    void createImageViews() {
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

    VkImageView createColorView(VkImage image, VkFormat format) {
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = format;
        ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &view));
        return view;
    }

    void createOffscreenImage(
        VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
        createImage(physicalDevice_,
                    device_,
                    swapchainExtent_.width,
                    swapchainExtent_.height,
                    format,
                    VK_IMAGE_TILING_OPTIMAL,
                    usage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    image,
                    memory);
        view = createColorView(image, format);
    }

    void createOffscreenImages() {
        const VkImageUsageFlags colorUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        createOffscreenImage(swapchainFormat_, colorUsage, opaqueImage_, opaqueMemory_, opaqueView_);
        createOffscreenImage(VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage, accumImage_, accumMemory_, accumView_);
        createOffscreenImage(VK_FORMAT_R8_UNORM, colorUsage, revealImage_, revealMemory_, revealView_);
    }

    void createSampler() {
        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &sampler_));
    }

    void createRenderPasses() {
        VkAttachmentDescription opaqueAtt{};
        opaqueAtt.format = swapchainFormat_;
        opaqueAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        opaqueAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        opaqueAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        opaqueAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        opaqueAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentReference opaqueRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription opaqueSp{};
        opaqueSp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        opaqueSp.colorAttachmentCount = 1;
        opaqueSp.pColorAttachments = &opaqueRef;
        VkSubpassDependency opaqueDep{};
        opaqueDep.srcSubpass = VK_SUBPASS_EXTERNAL;
        opaqueDep.dstSubpass = 0;
        opaqueDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        opaqueDep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        opaqueDep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo opaqueRpi{};
        opaqueRpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        opaqueRpi.attachmentCount = 1;
        opaqueRpi.pAttachments = &opaqueAtt;
        opaqueRpi.subpassCount = 1;
        opaqueRpi.pSubpasses = &opaqueSp;
        opaqueRpi.dependencyCount = 1;
        opaqueRpi.pDependencies = &opaqueDep;
        VK_CHECK(vkCreateRenderPass(device_, &opaqueRpi, nullptr, &opaqueRenderPass_));

        VkAttachmentDescription accumAtt{};
        accumAtt.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        accumAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        accumAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        accumAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        accumAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        accumAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentDescription revealAtt = accumAtt;
        revealAtt.format = VK_FORMAT_R8_UNORM;
        VkAttachmentDescription depthAtt{};
        depthAtt.format = depth_.format;
        depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference accumRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference revealRef{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkAttachmentReference oitColors[] = {accumRef, revealRef};
        VkSubpassDescription oitSp{};
        oitSp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        oitSp.colorAttachmentCount = 2;
        oitSp.pColorAttachments = oitColors;
        oitSp.pDepthStencilAttachment = &depthRef;
        std::array<VkAttachmentDescription, 3> oitAtts = {accumAtt, revealAtt, depthAtt};
        VkRenderPassCreateInfo oitRpi{};
        oitRpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        oitRpi.attachmentCount = static_cast<uint32_t>(oitAtts.size());
        oitRpi.pAttachments = oitAtts.data();
        oitRpi.subpassCount = 1;
        oitRpi.pSubpasses = &oitSp;
        oitRpi.dependencyCount = 1;
        oitRpi.pDependencies = &opaqueDep;
        VK_CHECK(vkCreateRenderPass(device_, &oitRpi, nullptr, &oitRenderPass_));

        VkAttachmentDescription swapAtt = opaqueAtt;
        swapAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkRenderPassCreateInfo compRpi{};
        compRpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        compRpi.attachmentCount = 1;
        compRpi.pAttachments = &swapAtt;
        compRpi.subpassCount = 1;
        compRpi.pSubpasses = &opaqueSp;
        compRpi.dependencyCount = 1;
        compRpi.pDependencies = &opaqueDep;
        VK_CHECK(vkCreateRenderPass(device_, &compRpi, nullptr, &compositeRenderPass_));
    }

    void createFramebuffers() {
        VkImageView opaqueAtt[] = {opaqueView_};
        VkFramebufferCreateInfo opaqueFb{};
        opaqueFb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        opaqueFb.renderPass = opaqueRenderPass_;
        opaqueFb.attachmentCount = 1;
        opaqueFb.pAttachments = opaqueAtt;
        opaqueFb.width = swapchainExtent_.width;
        opaqueFb.height = swapchainExtent_.height;
        opaqueFb.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &opaqueFb, nullptr, &opaqueFramebuffer_));

        VkImageView oitAtt[] = {accumView_, revealView_, depth_.view};
        VkFramebufferCreateInfo oitFb = opaqueFb;
        oitFb.renderPass = oitRenderPass_;
        oitFb.attachmentCount = 3;
        oitFb.pAttachments = oitAtt;
        VK_CHECK(vkCreateFramebuffer(device_, &oitFb, nullptr, &oitFramebuffer_));

        compositeFramebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            VkImageView att[] = {swapchainImageViews_[i]};
            VkFramebufferCreateInfo fb = opaqueFb;
            fb.renderPass = compositeRenderPass_;
            fb.pAttachments = att;
            VK_CHECK(vkCreateFramebuffer(device_, &fb, nullptr, &compositeFramebuffers_[i]));
        }
    }

    void createDescriptorLayouts() {
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

        std::array<VkDescriptorSetLayoutBinding, 3> compBindings{};
        for (uint32_t i = 0; i < 3; ++i) {
            compBindings[i].binding = i;
            compBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            compBindings[i].descriptorCount = 1;
            compBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo compLayout{};
        compLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        compLayout.bindingCount = static_cast<uint32_t>(compBindings.size());
        compLayout.pBindings = compBindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &compLayout, nullptr, &compositeSetLayout_));
        VkPipelineLayoutCreateInfo compPl{};
        compPl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        compPl.setLayoutCount = 1;
        compPl.pSetLayouts = &compositeSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &compPl, nullptr, &compositePipeLayout_));
    }

    VkPipeline buildGraphicsPipeline(VkRenderPass renderPass,
                                     VkPipelineLayout layout,
                                     const char* vertSpv,
                                     const char* fragSpv,
                                     bool depthTest,
                                     bool mrtBlend) {
        VkShaderModule vert = createShaderModuleFromFile(device_, vertSpv);
        VkShaderModule frag = createShaderModuleFromFile(device_, fragSpv);
        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_VERTEX_BIT,
                     vert,
                     "main",
                     nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_FRAGMENT_BIT,
                     frag,
                     "main",
                     nullptr};
        VkVertexInputBindingDescription bind{0, sizeof(OitVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 3> attrs{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(OitVertex, pos)};
        attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(OitVertex, color)};
        attrs[2] = {2, 0, VK_FORMAT_R32_SFLOAT, offsetof(OitVertex, alpha)};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
        vi.pVertexAttributeDescriptions = attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                  nullptr,
                                                  0,
                                                  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                  VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
        ds.depthWriteEnable = depthTest ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;
        std::array<VkPipelineColorBlendAttachmentState, 2> cbas{};
        if (mrtBlend) {
            cbas[0] = {VK_TRUE,
                       VK_BLEND_FACTOR_ONE,
                       VK_BLEND_FACTOR_ONE,
                       VK_BLEND_OP_ADD,
                       VK_BLEND_FACTOR_ONE,
                       VK_BLEND_FACTOR_ONE,
                       VK_BLEND_OP_ADD,
                       VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                           VK_COLOR_COMPONENT_A_BIT};
            cbas[1] = {VK_TRUE,
                       VK_BLEND_FACTOR_ZERO,
                       VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                       VK_BLEND_OP_ADD,
                       VK_BLEND_FACTOR_ZERO,
                       VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                       VK_BLEND_OP_ADD,
                       VK_COLOR_COMPONENT_R_BIT};
        } else {
            cbas[0].colorWriteMask = 0xF;
        }
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = mrtBlend ? 2u : 1u;
        cb.pAttachments = cbas.data();
        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};
        dynS.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynS.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        dynS.pDynamicStates = dyn.data();
        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount = 2;
        pi.pStages = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vs;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &cb;
        pi.pDynamicState = &dynS;
        pi.layout = layout;
        pi.renderPass = renderPass;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        return pipeline;
    }

    VkPipeline buildFullscreenPipeline(VkRenderPass renderPass, VkPipelineLayout layout, const char* fragSpv) {
        VkShaderModule vert = createShaderModuleFromFile(device_, "tonemap.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, fragSpv);
        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_VERTEX_BIT,
                     vert,
                     "main",
                     nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_FRAGMENT_BIT,
                     frag,
                     "main",
                     nullptr};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                  nullptr,
                                                  0,
                                                  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                  VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
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
        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount = 2;
        pi.pStages = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vs;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pColorBlendState = &cb;
        pi.pDynamicState = &dynS;
        pi.layout = layout;
        pi.renderPass = renderPass;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        return pipeline;
    }

    void createPipelines() {
        opaquePipeline_ = buildGraphicsPipeline(
            opaqueRenderPass_, scenePipeLayout_, "oit.vert.spv", "alpha_blend.frag.spv", false, false);
        oitPipeline_ =
            buildGraphicsPipeline(oitRenderPass_, scenePipeLayout_, "oit.vert.spv", "oit_accum.frag.spv", true, true);
        compositePipeline_ =
            buildFullscreenPipeline(compositeRenderPass_, compositePipeLayout_, "oit_composite.frag.spv");
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
    }

    void createVertexBuffer() {
        std::vector<OitVertex> vertices;
        auto append = [&](const std::vector<OitVertex>& quad) {
            vertices.insert(vertices.end(), quad.begin(), quad.end());
        };
        append(buildQuad({0.0f, 0.0f, 0.0f}, {6.0f, 4.0f}, {0.12f, 0.12f, 0.18f}, 1.0f));
        opaqueVertexCount_ = static_cast<uint32_t>(vertices.size());
        append(buildQuad({-0.6f, 0.0f, 0.3f}, {1.4f, 1.4f}, {1.0f, 0.2f, 0.2f}, 0.45f));
        append(buildQuad({0.0f, 0.2f, 0.15f}, {1.6f, 1.6f}, {0.2f, 0.85f, 0.35f}, 0.55f));
        append(buildQuad({0.65f, -0.1f, 0.25f}, {1.5f, 1.5f}, {0.25f, 0.45f, 1.0f}, 0.50f));
        append(buildQuad({0.1f, -0.3f, 0.35f}, {1.7f, 1.2f}, {1.0f, 0.85f, 0.15f}, 0.40f));
        transparentVertexCount_ = static_cast<uint32_t>(vertices.size()) - opaqueVertexCount_;
        const VkDeviceSize bufferSize = sizeof(OitVertex) * vertices.size();
        createBuffer(physicalDevice_,
                     device_,
                     bufferSize,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_,
                     vertexMemory_);
        void* mapped = nullptr;
        VK_CHECK(vkMapMemory(device_, vertexMemory_, 0, bufferSize, 0, &mapped));
        std::memcpy(mapped, vertices.data(), static_cast<size_t>(bufferSize));
        vkUnmapMemory(device_, vertexMemory_);
    }

    void createSceneUBOs() {
        sceneUBOs_.resize(MAX_FRAMES);
        sceneUBOMemories_.resize(MAX_FRAMES);
        sceneUBOMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physicalDevice_,
                         device_,
                         sizeof(SceneUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         sceneUBOs_[i],
                         sceneUBOMemories_[i]);
            VK_CHECK(vkMapMemory(device_, sceneUBOMemories_[i], 0, sizeof(SceneUBO), 0, &sceneUBOMapped_[i]));
        }
    }

    void createDescriptorPoolAndSets() {
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(MAX_FRAMES)},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
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
        ai.pSetLayouts = &compositeSetLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &compositeDescSet_));
        VkDescriptorImageInfo images[3] = {
            {sampler_, opaqueView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {sampler_, accumView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {sampler_, revealView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        };
        VkWriteDescriptorSet writes[3] = {};
        for (uint32_t i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = compositeDescSet_;
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].descriptorCount = 1;
            writes[i].pImageInfo = &images[i];
        }
        vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);
    }

    void createCommandBuffers() {
        commandBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()));
    }

    void createSyncObjects() {
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

    void updateUBO(uint32_t frameIndex, float time) {
        SceneUBO ubo{};
        ubo.view = interactive_.camera().viewMatrix();
        const float aspect = static_cast<float>(swapchainExtent_.width) / static_cast<float>(swapchainExtent_.height);
        ubo.projection = interactive_.camera().projectionMatrix(aspect, 45.0f, 0.1f, 20.0f);
        ubo.model = glm::rotate(glm::mat4(1.0f), time * 0.25f, glm::vec3(0.0f, 1.0f, 0.0f));
        std::memcpy(sceneUBOMapped_[frameIndex], &ubo, sizeof(ubo));
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex, float time) {
        updateUBO(currentFrame_, time);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, currentFrame_);
        VkViewport vp{0.0f,
                      0.0f,
                      static_cast<float>(swapchainExtent_.width),
                      static_cast<float>(swapchainExtent_.height),
                      0.0f,
                      1.0f};
        VkRect2D sc{{0, 0}, swapchainExtent_};
        VkDeviceSize offset = 0;
        VkBuffer vbo = vertexBuffer_;

        VkClearValue opaqueClear = {{{0.08f, 0.08f, 0.12f, 1.0f}}};
        VkRenderPassBeginInfo opaqueRp{};
        opaqueRp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        opaqueRp.renderPass = opaqueRenderPass_;
        opaqueRp.framebuffer = opaqueFramebuffer_;
        opaqueRp.renderArea = sc;
        opaqueRp.clearValueCount = 1;
        opaqueRp.pClearValues = &opaqueClear;
        vkCmdBeginRenderPass(cmd, &opaqueRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, opaquePipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeLayout_, 0, 1, &sceneDescSets_[currentFrame_], 0, nullptr);
        vkCmdBindVertexBuffers(cmd, 0, 1, &vbo, &offset);
        vkCmdDraw(cmd, opaqueVertexCount_, 1, 0, 0);
        vkCmdEndRenderPass(cmd);

        std::array<VkClearValue, 3> oitClears{};
        oitClears[0] = {{{0.0f, 0.0f, 0.0f, 0.0f}}};
        oitClears[1] = {{{1.0f, 0.0f, 0.0f, 0.0f}}};
        oitClears[2].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo oitRp{};
        oitRp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        oitRp.renderPass = oitRenderPass_;
        oitRp.framebuffer = oitFramebuffer_;
        oitRp.renderArea = sc;
        oitRp.clearValueCount = static_cast<uint32_t>(oitClears.size());
        oitRp.pClearValues = oitClears.data();
        vkCmdBeginRenderPass(cmd, &oitRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, oitPipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeLayout_, 0, 1, &sceneDescSets_[currentFrame_], 0, nullptr);
        vkCmdBindVertexBuffers(cmd, 0, 1, &vbo, &offset);
        vkCmdDraw(cmd, transparentVertexCount_, 1, opaqueVertexCount_, 0);
        vkCmdEndRenderPass(cmd);

        VkClearValue swapClear = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        VkRenderPassBeginInfo compRp{};
        compRp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        compRp.renderPass = compositeRenderPass_;
        compRp.framebuffer = compositeFramebuffers_[imageIndex];
        compRp.renderArea = sc;
        compRp.clearValueCount = 1;
        compRp.pClearValues = &swapClear;
        vkCmdBeginRenderPass(cmd, &compRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeLayout_, 0, 1, &compositeDescSet_, 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);
        interactive_.endGpuSection(cmd, currentFrame_);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex = 0;
        VkResult result = vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX, imageAvailableSems_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
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

    void cleanupOffscreen() {
        if (opaqueView_)
            vkDestroyImageView(device_, opaqueView_, nullptr);
        if (accumView_)
            vkDestroyImageView(device_, accumView_, nullptr);
        if (revealView_)
            vkDestroyImageView(device_, revealView_, nullptr);
        if (opaqueImage_)
            vkDestroyImage(device_, opaqueImage_, nullptr);
        if (accumImage_)
            vkDestroyImage(device_, accumImage_, nullptr);
        if (revealImage_)
            vkDestroyImage(device_, revealImage_, nullptr);
        if (opaqueMemory_)
            vkFreeMemory(device_, opaqueMemory_, nullptr);
        if (accumMemory_)
            vkFreeMemory(device_, accumMemory_, nullptr);
        if (revealMemory_)
            vkFreeMemory(device_, revealMemory_, nullptr);
        opaqueView_ = accumView_ = revealView_ = VK_NULL_HANDLE;
        opaqueImage_ = accumImage_ = revealImage_ = VK_NULL_HANDLE;
        opaqueMemory_ = accumMemory_ = revealMemory_ = VK_NULL_HANDLE;
    }

    void recreateSwapchain() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        for (auto fb : compositeFramebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroyFramebuffer(device_, opaqueFramebuffer_, nullptr);
        vkDestroyFramebuffer(device_, oitFramebuffer_, nullptr);
        destroyDepthResources(device_, depth_);
        cleanupOffscreen();
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physicalDevice_, device_, swapchainExtent_);
        createOffscreenImages();
        createFramebuffers();
        interactive_.onSwapchainRecreated(
            compositeRenderPass_, swapchainFormat_, static_cast<uint32_t>(swapchainImages_.size()));
    }

    void mainLoop() {
        std::cout << "🎨 重叠透明四边形演示 OIT。ESC 退出。\n";
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

    void cleanup() {
        vkDestroyPipeline(device_, compositePipeline_, nullptr);
        vkDestroyPipeline(device_, oitPipeline_, nullptr);
        vkDestroyPipeline(device_, opaquePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, compositePipeLayout_, nullptr);
        vkDestroyPipelineLayout(device_, scenePipeLayout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, compositeSetLayout_, nullptr);
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
        for (auto fb : compositeFramebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyFramebuffer(device_, oitFramebuffer_, nullptr);
        vkDestroyFramebuffer(device_, opaqueFramebuffer_, nullptr);
        for (auto iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroyRenderPass(device_, compositeRenderPass_, nullptr);
        vkDestroyRenderPass(device_, oitRenderPass_, nullptr);
        vkDestroyRenderPass(device_, opaqueRenderPass_, nullptr);
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

int main() {
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << " 第48章：Weighted Blended OIT\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    Ch48App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
