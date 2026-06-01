/**
 * @file ch24_postprocess.cpp
 * @brief 第24章：后处理效果（HDR + Bloom + Tone Mapping + Gamma Correction）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【后处理渲染管线】
 *
 *  ┌──────────────────────────────────────────────────────────────────┐
 *  │  Pass 1: Scene Render → HDR Framebuffer（RGBA16F）               │
 *  │    - 场景颜色可超过 1.0（表示高亮区域）                          │
 *  │    - 同时输出 Bright Buffer（亮度 > 1.0 的区域）                 │
 *  │                                                                  │
 *  │  Pass 2: Gaussian Blur（Compute Shader）                         │
 *  │    - 对 Bright Buffer 进行水平 + 垂直高斯模糊                    │
 *  │    - 模糊半径控制泛光范围                                         │
 *  │                                                                  │
 *  │  Pass 3: Composite + Tone Mapping + Gamma（全屏四边形）          │
 *  │    - HDR 颜色 + Bloom 叠加                                       │
 *  │    - ACES Filmic Tone Mapping：HDR → LDR                        │
 *  │    - Gamma 校正：线性空间 → sRGB 显示空间                        │
 *  └──────────────────────────────────────────────────────────────────┘
 *
 * 【HDR（High Dynamic Range）渲染】
 *
 *  传统渲染：颜色固定在 [0,1]，光照强的地方一片白（过曝）
 *  HDR 渲染：颜色范围 [0, ∞)，保留高亮细节
 *  RGBA16F：每通道 16-bit 浮点，足够表示 HDR 范围
 *
 * 【Bloom 泛光】
 *
 *  物理现实中，强光源在相机/眼睛中产生"泛光"效果。
 *  实现：提取超亮像素 → 高斯模糊 → 叠加回原图
 *  高斯模糊：沿两个轴各一遍（利用高斯核的可分离性）
 *
 * 【Tone Mapping】
 *
 *  将无限范围的 HDR 颜色映射到 [0,1] 的方法：
 *    Reinhard:       简单，但过亮区域对比度损失
 *    Uncharted 2:    游戏行业常用，S形曲线
 *    ACES Filmic:    行业标准，最接近电影胶片效果（本章使用）
 *
 * 【Gamma Correction】
 *
 *  显示器假设输入是 gamma=2.2 的非线性空间（sRGB）。
 *  但渲染在线性空间进行，最终输出需要 pow(color, 1/2.2)。
 *  如果不做 gamma 校正：颜色偏暗，阴影细节丢失。
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vulkan_tutorial/utils.hpp>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;

struct SceneVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
};

// 亮度很高的场景（刻意超过1.0，产生 Bloom 效果）
static const std::vector<SceneVertex> SCENE = {
    // 地面（普通亮度）
    {{-4, -0.5f, -4}, {0, 1, 0}, {0.4f, 0.4f, 0.5f}},
    {{4, -0.5f, -4}, {0, 1, 0}, {0.4f, 0.4f, 0.5f}},
    {{4, -0.5f, 4}, {0, 1, 0}, {0.4f, 0.4f, 0.5f}},
    {{-4, -0.5f, -4}, {0, 1, 0}, {0.4f, 0.4f, 0.5f}},
    {{4, -0.5f, 4}, {0, 1, 0}, {0.4f, 0.4f, 0.5f}},
    {{-4, -0.5f, 4}, {0, 1, 0}, {0.4f, 0.4f, 0.5f}},
    // 中心高亮立面（颜色 * brightness 后远超 1.0 → 产生 Bloom）
    {{-0.5f, -0.5f, 0}, {0, 0, -1}, {3.0f, 2.5f, 0.2f}}, // 极亮黄色
    {{0.5f, -0.5f, 0}, {0, 0, -1}, {3.0f, 2.5f, 0.2f}},
    {{0.5f, 1.5f, 0}, {0, 0, -1}, {3.0f, 2.5f, 0.2f}},
    {{-0.5f, -0.5f, 0}, {0, 0, -1}, {3.0f, 2.5f, 0.2f}},
    {{0.5f, 1.5f, 0}, {0, 0, -1}, {3.0f, 2.5f, 0.2f}},
    {{-0.5f, 1.5f, 0}, {0, 0, -1}, {3.0f, 2.5f, 0.2f}},
    // 左侧蓝色高亮柱
    {{-2.0f, -0.5f, 0}, {1, 0, 0}, {0.2f, 1.0f, 3.5f}},
    {{-1.5f, -0.5f, 0}, {1, 0, 0}, {0.2f, 1.0f, 3.5f}},
    {{-1.5f, 2.0f, 0}, {1, 0, 0}, {0.2f, 1.0f, 3.5f}},
    {{-2.0f, -0.5f, 0}, {1, 0, 0}, {0.2f, 1.0f, 3.5f}},
    {{-1.5f, 2.0f, 0}, {1, 0, 0}, {0.2f, 1.0f, 3.5f}},
    {{-2.0f, 2.0f, 0}, {1, 0, 0}, {0.2f, 1.0f, 3.5f}},
};

struct SceneUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::vec4 lightDir;
    alignas(16) glm::vec4 lightColor;
    float exposure;
};

struct BlurPushConstants {
    int horizontal;
    float strength;
};
struct ToneMapPushConstants {
    float exposure;
    float bloomStrength;
    float gamma;
};

class Ch24App {
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

    // ─── Pass 1：HDR 场景渲染 ─────────────────────────────────────────────
    VkImage hdrImage_ = VK_NULL_HANDLE; // HDR 颜色（RGBA16F）
    VkDeviceMemory hdrMemory_ = VK_NULL_HANDLE;
    VkImageView hdrView_ = VK_NULL_HANDLE;
    VkImage brightImage_ = VK_NULL_HANDLE; // 超亮像素（用于 Bloom）
    VkDeviceMemory brightMemory_ = VK_NULL_HANDLE;
    VkImageView brightView_ = VK_NULL_HANDLE;
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;
    VkRenderPass sceneRenderPass_ = VK_NULL_HANDLE;
    VkFramebuffer sceneFramebuffer_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sceneSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout scenePipeLayout_ = VK_NULL_HANDLE;
    VkPipeline scenePipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool sceneDescPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneDescSets_;
    std::vector<VkBuffer> sceneUBOs_;
    std::vector<VkDeviceMemory> sceneUBOMemories_;
    std::vector<void*> sceneUBOMapped_;
    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;

    // ─── Pass 2：Bloom Blur（Compute） ───────────────────────────────────
    VkImage blurImage_[2] = {}; // 乒乓缓冲（水平/垂直交替模糊）
    VkDeviceMemory blurMemory_[2] = {};
    VkImageView blurView_[2] = {};
    VkDescriptorSetLayout blurSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout blurPipeLayout_ = VK_NULL_HANDLE;
    VkPipeline blurPipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool blurDescPool_ = VK_NULL_HANDLE;
    VkDescriptorSet blurDescSets_[4] = {}; // [pass0_read, pass0_write, pass1_read, pass1_write]

    // ─── Pass 3：Tone Mapping + 呈现 ─────────────────────────────────────
    VkRenderPass tonemapRenderPass_ = VK_NULL_HANDLE;
    VkSampler hdrSampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout tonemapSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout tonemapPipeLayout_ = VK_NULL_HANDLE;
    VkPipeline tonemapPipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool tonemapDescPool_ = VK_NULL_HANDLE;
    VkDescriptorSet tonemapDescSet_ = VK_NULL_HANDLE;

    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkFramebuffer> tonemapFramebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    QueueFamilyIndices queueIndices_;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;

    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch24 - HDR + Bloom + Tone Mapping", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch24App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        depthFormat_ = findDepthFormat();
        createHDRResources();  // HDR + Bright + Depth 图像
        createBlurResources(); // 乒乓模糊缓冲
        createSampler();
        createSceneRenderPass();
        createScenePipeline();
        createSceneFramebuffer();
        createTonemapRenderPass();
        createTonemapPipeline();
        createTonemapFramebuffers();
        createBlurPipeline();
        createCommandPool();
        createVertexBuffer();
        createSceneUBOs();
        createDescriptorPools();
        createSceneDescriptorSets();
        createBlurDescriptorSets();
        createTonemapDescriptorSet();
        createCommandBuffers();
        createSyncObjects();
        std::cout << "\n✅ HDR + Bloom + Tone Mapping 初始化完成！\n";
        std::cout << "🎨 HDR 格式：RGBA16F（亮度可超过 1.0）\n";
        std::cout << "✨ Bloom：5-tap Gaussian Blur × 2 遍（水平+垂直）\n";
        std::cout << "🎬 Tone Mapping：ACES Filmic\n";
        std::cout << "🔆 Gamma Correction：2.2\n";
    }

    void createHDRResources() {
        auto create = [this](VkFormat fmt,
                             VkImageUsageFlags usage,
                             VkImage& img,
                             VkDeviceMemory& mem,
                             VkImageView& view,
                             VkImageAspectFlags aspect) {
            VkImageCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ci.imageType = VK_IMAGE_TYPE_2D;
            ci.extent = {swapchainExtent_.width, swapchainExtent_.height, 1};
            ci.mipLevels = 1;
            ci.arrayLayers = 1;
            ci.format = fmt;
            ci.tiling = VK_IMAGE_TILING_OPTIMAL;
            ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ci.usage = usage;
            ci.samples = VK_SAMPLE_COUNT_1_BIT;
            VK_CHECK(vkCreateImage(device_, &ci, nullptr, &img));
            VkMemoryRequirements mr;
            vkGetImageMemoryRequirements(device_, img, &mr);
            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = mr.size;
            ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &mem));
            VK_CHECK(vkBindImageMemory(device_, img, mem, 0));
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = img;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = fmt;
            vci.subresourceRange = {aspect, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &view));
        };

        VkImageUsageFlags colorUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        create(VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage, hdrImage_, hdrMemory_, hdrView_, VK_IMAGE_ASPECT_COLOR_BIT);
        create(VK_FORMAT_R16G16B16A16_SFLOAT,
               colorUsage | VK_IMAGE_USAGE_STORAGE_BIT,
               brightImage_,
               brightMemory_,
               brightView_,
               VK_IMAGE_ASPECT_COLOR_BIT);
        create(depthFormat_,
               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
               depthImage_,
               depthMemory_,
               depthView_,
               VK_IMAGE_ASPECT_DEPTH_BIT);

        std::cout << "✅ HDR 缓冲：RGBA16F (hdr) + RGBA16F (bright) + Depth\n";
    }

    void createBlurResources() {
        for (int i = 0; i < 2; ++i) {
            VkImageCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ci.imageType = VK_IMAGE_TYPE_2D;
            ci.extent = {swapchainExtent_.width, swapchainExtent_.height, 1};
            ci.mipLevels = 1;
            ci.arrayLayers = 1;
            ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            ci.tiling = VK_IMAGE_TILING_OPTIMAL;
            ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ci.samples = VK_SAMPLE_COUNT_1_BIT;
            VK_CHECK(vkCreateImage(device_, &ci, nullptr, &blurImage_[i]));
            VkMemoryRequirements mr;
            vkGetImageMemoryRequirements(device_, blurImage_[i], &mr);
            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = mr.size;
            ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &blurMemory_[i]));
            VK_CHECK(vkBindImageMemory(device_, blurImage_[i], blurMemory_[i], 0));
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = blurImage_[i];
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &blurView_[i]));
        }
    }

    void createSampler() {
        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &hdrSampler_));
    }

    void createSceneRenderPass() {
        // MRT：HDR 颜色 + Bright 颜色（超亮部分） + 深度
        VkAttachmentDescription hdrAtt{};
        hdrAtt.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        hdrAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        hdrAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        hdrAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        hdrAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        hdrAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        hdrAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        hdrAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentDescription brightAtt = hdrAtt;
        brightAtt.finalLayout = VK_IMAGE_LAYOUT_GENERAL; // Storage + Sampled
        VkAttachmentDescription depthAtt{};
        depthAtt.format = depthFormat_;
        depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference hdrRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference brightRef{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkAttachmentReference colorRefs[] = {hdrRef, brightRef};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 2;
        sp.pColorAttachments = colorRefs;
        sp.pDepthStencilAttachment = &depthRef;
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        std::array<VkAttachmentDescription, 3> atts = {hdrAtt, brightAtt, depthAtt};
        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = 3;
        rpi.pAttachments = atts.data();
        rpi.subpassCount = 1;
        rpi.pSubpasses = &sp;
        rpi.dependencyCount = 1;
        rpi.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &sceneRenderPass_));
    }

    void createSceneFramebuffer() {
        std::array<VkImageView, 3> att = {hdrView_, brightView_, depthView_};
        VkFramebufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass = sceneRenderPass_;
        ci.attachmentCount = 3;
        ci.pAttachments = att.data();
        ci.width = swapchainExtent_.width;
        ci.height = swapchainExtent_.height;
        ci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &sceneFramebuffer_));
    }

    void createScenePipeline() {
        VkShaderModule vert = createShaderModuleFromFile(device_, "hdr_scene.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "hdr_scene.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{};
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
        VkVertexInputBindingDescription bind{0, sizeof(SceneVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 3> attrs{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
        attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12};
        attrs[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, 24};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = 3;
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
        rs.lineWidth = 1.0f;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;
        // MRT：2 个颜色附件各自的混合配置
        std::array<VkPipelineColorBlendAttachmentState, 2> cbas{};
        for (auto& c : cbas) {
            c.colorWriteMask = 0xf;
            c.blendEnable = VK_FALSE;
        }
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 2;
        cb.pAttachments = cbas.data();
        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};
        dynS.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynS.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        dynS.pDynamicStates = dyn.data();
        VkDescriptorSetLayoutBinding b{0,
                                       VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       1,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       nullptr};
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 1;
        dlci.pBindings = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dlci, nullptr, &sceneSetLayout_));
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &sceneSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &scenePipeLayout_));
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
        pi.layout = scenePipeLayout_;
        pi.renderPass = sceneRenderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &scenePipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        std::cout << "✅ HDR 场景管线创建（MRT: hdr + bright，含深度测试）\n";
    }

    void createBlurPipeline() {
        VkShaderModule comp = createShaderModuleFromFile(device_, "bloom_blur.comp.spv");
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = comp;
        stage.pName = "main";
        // 布局：2个 Storage Images（输入+输出，交替使用）
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 2;
        dlci.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dlci, nullptr, &blurSetLayout_));
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(BlurPushConstants);
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &blurSetLayout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &blurPipeLayout_));
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage = stage;
        ci.layout = blurPipeLayout_;
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &blurPipeline_));
        vkDestroyShaderModule(device_, comp, nullptr);
        std::cout << "✅ Bloom 高斯模糊管线创建（Compute, 5-tap）\n";
    }

    void createTonemapRenderPass() {
        VkAttachmentDescription ca{};
        ca.format = swapchainImageFormat_;
        ca.samples = VK_SAMPLE_COUNT_1_BIT;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ca.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ca.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ca.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ca.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference cr{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &cr;
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = 1;
        rpi.pAttachments = &ca;
        rpi.subpassCount = 1;
        rpi.pSubpasses = &sp;
        rpi.dependencyCount = 1;
        rpi.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &tonemapRenderPass_));
    }

    void createTonemapPipeline() {
        VkShaderModule vert = createShaderModuleFromFile(device_, "tonemap.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "tonemap.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
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
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.lineWidth = 1.0f;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};
        dynS.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynS.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        dynS.pDynamicStates = dyn.data();
        // 2 个纹理采样器（HDR + Bloom）
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 2;
        dlci.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dlci, nullptr, &tonemapSetLayout_));
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(ToneMapPushConstants);
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &tonemapSetLayout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &tonemapPipeLayout_));
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
        pi.layout = tonemapPipeLayout_;
        pi.renderPass = tonemapRenderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &tonemapPipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        std::cout << "✅ Tone Mapping 管线创建（ACES Filmic + Gamma 2.2）\n";
    }

    void createTonemapFramebuffers() {
        tonemapFramebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            VkImageView att[] = {swapchainImageViews_[i]};
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = tonemapRenderPass_;
            ci.attachmentCount = 1;
            ci.pAttachments = att;
            ci.width = swapchainExtent_.width;
            ci.height = swapchainExtent_.height;
            ci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &tonemapFramebuffers_[i]));
        }
    }

    void createDescriptorPools() {
        // 场景 UBO 描述符池
        VkDescriptorPoolSize ps0{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(MAX_FRAMES)};
        VkDescriptorPoolCreateInfo poolCI0{};
        poolCI0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI0.poolSizeCount = 1;
        poolCI0.pPoolSizes = &ps0;
        poolCI0.maxSets = MAX_FRAMES;
        VK_CHECK(vkCreateDescriptorPool(device_, &poolCI0, nullptr, &sceneDescPool_));
        // Blur Storage Images 描述符池
        VkDescriptorPoolSize ps1{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8};
        VkDescriptorPoolCreateInfo poolCI1{};
        poolCI1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI1.poolSizeCount = 1;
        poolCI1.pPoolSizes = &ps1;
        poolCI1.maxSets = 4;
        VK_CHECK(vkCreateDescriptorPool(device_, &poolCI1, nullptr, &blurDescPool_));
        // Tonemap Sampler 描述符池
        VkDescriptorPoolSize ps2{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2};
        VkDescriptorPoolCreateInfo poolCI2{};
        poolCI2.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI2.poolSizeCount = 1;
        poolCI2.pPoolSizes = &ps2;
        poolCI2.maxSets = 1;
        VK_CHECK(vkCreateDescriptorPool(device_, &poolCI2, nullptr, &tonemapDescPool_));
    }

    void createSceneUBOs() {
        sceneUBOs_.resize(MAX_FRAMES);
        sceneUBOMemories_.resize(MAX_FRAMES);
        sceneUBOMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(sizeof(SceneUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         sceneUBOs_[i],
                         sceneUBOMemories_[i]);
            vkMapMemory(device_, sceneUBOMemories_[i], 0, sizeof(SceneUBO), 0, &sceneUBOMapped_[i]);
        }
    }

    void createSceneDescriptorSets() {
        std::vector<VkDescriptorSetLayout> lays(MAX_FRAMES, sceneSetLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = sceneDescPool_;
        ai.descriptorSetCount = MAX_FRAMES;
        ai.pSetLayouts = lays.data();
        sceneDescSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, sceneDescSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = sceneUBOs_[i];
            bi.offset = 0;
            bi.range = sizeof(SceneUBO);
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = sceneDescSets_[i];
            w.dstBinding = 0;
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &bi;
            vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
        }
    }

    void createBlurDescriptorSets() {
        // 4 个描述符集：
        // [0]: Pass0 读 brightImage_, 写 blurImage_[0]
        // [1]: Pass1 读 blurImage_[0], 写 blurImage_[1]
        std::array<VkDescriptorSetLayout, 4> lays{blurSetLayout_, blurSetLayout_, blurSetLayout_, blurSetLayout_};
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = blurDescPool_;
        ai.descriptorSetCount = 4;
        ai.pSetLayouts = lays.data();
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, blurDescSets_));
        auto updateBlurSet = [this](VkDescriptorSet set, VkImageView in, VkImageView out) {
            VkDescriptorImageInfo ii0{VK_NULL_HANDLE, in, VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo ii1{VK_NULL_HANDLE, out, VK_IMAGE_LAYOUT_GENERAL};
            std::array<VkWriteDescriptorSet, 2> ws{};
            ws[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                     nullptr,
                     set,
                     0,
                     0,
                     1,
                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                     &ii0,
                     nullptr,
                     nullptr};
            ws[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                     nullptr,
                     set,
                     1,
                     0,
                     1,
                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                     &ii1,
                     nullptr,
                     nullptr};
            vkUpdateDescriptorSets(device_, 2, ws.data(), 0, nullptr);
        };
        updateBlurSet(blurDescSets_[0], brightView_,
                      blurView_[0]); // Bright → BlurH
        updateBlurSet(blurDescSets_[1], blurView_[0],
                      blurView_[1]); // BlurH  → BlurV
        updateBlurSet(blurDescSets_[2], blurView_[1],
                      blurView_[0]); // 额外迭代（可选）
        updateBlurSet(blurDescSets_[3], blurView_[0], blurView_[1]);
    }

    void createTonemapDescriptorSet() {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = tonemapDescPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &tonemapSetLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &tonemapDescSet_));
        // HDR 颜色缓冲 + 最终模糊后的 Bloom 缓冲
        VkDescriptorImageInfo hdrII{hdrSampler_, hdrView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo bloomII{hdrSampler_, blurView_[1], VK_IMAGE_LAYOUT_GENERAL};
        std::array<VkWriteDescriptorSet, 2> ws{};
        ws[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 tonemapDescSet_,
                 0,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 &hdrII,
                 nullptr,
                 nullptr};
        ws[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 tonemapDescSet_,
                 1,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 &bloomII,
                 nullptr,
                 nullptr};
        vkUpdateDescriptorSets(device_, 2, ws.data(), 0, nullptr);
    }

    void updateSceneUBO(uint32_t frame) {
        static auto start = std::chrono::high_resolution_clock::now();
        float t = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();
        SceneUBO ubo{};
        ubo.model = glm::mat4(1.0f);
        ubo.view =
            glm::lookAt(glm::vec3(5 * sin(t * 0.2f), 3, 5 * cos(t * 0.2f)), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        ubo.projection =
            glm::perspective(glm::radians(60.0f), (float)swapchainExtent_.width / swapchainExtent_.height, 0.1f, 20.0f);
        ubo.projection[1][1] *= -1;
        ubo.lightDir = glm::vec4(glm::normalize(glm::vec3(-1, -2, -1)), 0);
        ubo.lightColor = glm::vec4(1.0f);
        ubo.exposure = 1.5f;
        std::memcpy(sceneUBOMapped_[frame], &ubo, sizeof(ubo));
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        // ════════════════════════════════════════════════════════════════════
        // Pass 1: HDR 场景渲染（MRT: hdr + bright）
        // ════════════════════════════════════════════════════════════════════
        {
            std::array<VkClearValue, 3> clears{};
            clears[0].color.float32[3] = 1.0f; // HDR buffer → 黑色
            clears[1].color.float32[3] = 1.0f; // Bright buffer → 黑色
            clears[2].depthStencil = {1.0f, 0};
            VkRenderPassBeginInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp.renderPass = sceneRenderPass_;
            rp.framebuffer = sceneFramebuffer_;
            rp.renderArea = {{0, 0}, swapchainExtent_};
            rp.clearValueCount = 3;
            rp.pClearValues = clears.data();
            vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeline_);
            VkViewport vp{0, 0, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0, 1};
            vkCmdSetViewport(cmd, 0, 1, &vp);
            VkRect2D sc{{0, 0}, swapchainExtent_};
            vkCmdSetScissor(cmd, 0, 1, &sc);
            VkBuffer vb[] = {vertexBuffer_};
            VkDeviceSize off[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
            vkCmdBindDescriptorSets(cmd,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    scenePipeLayout_,
                                    0,
                                    1,
                                    &sceneDescSets_[currentFrame_],
                                    0,
                                    nullptr);
            vkCmdDraw(cmd, static_cast<uint32_t>(SCENE.size()), 1, 0, 0);
            vkCmdEndRenderPass(cmd);
        }

        // Bright Image 转换到 GENERAL 布局（Compute 需要 Storage Image）
        // （已在 RenderPass finalLayout 中处理）

        // ════════════════════════════════════════════════════════════════════
        // Pass 2: Bloom Blur（2 遍 Gaussian Blur）
        // ════════════════════════════════════════════════════════════════════
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, blurPipeline_);
        uint32_t gx = (swapchainExtent_.width + 15) / 16;
        uint32_t gy = (swapchainExtent_.height + 15) / 16;

        // 水平模糊：Bright → BlurH
        {
            BlurPushConstants pc{1, 1.0f};
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_COMPUTE, blurPipeLayout_, 0, 1, &blurDescSets_[0], 0, nullptr);
            vkCmdPushConstants(cmd, blurPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, gx, gy, 1);
        }
        // Barrier：等 Pass0 写完，Pass1 才能读
        VkImageMemoryBarrier blurBarrier{};
        blurBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        blurBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        blurBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        blurBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        blurBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        blurBarrier.image = blurImage_[0];
        blurBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        blurBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        blurBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &blurBarrier);

        // 垂直模糊：BlurH → BlurV
        {
            BlurPushConstants pc{0, 1.0f};
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_COMPUTE, blurPipeLayout_, 0, 1, &blurDescSets_[1], 0, nullptr);
            vkCmdPushConstants(cmd, blurPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, gx, gy, 1);
        }
        // Barrier：等 Blur 写完再给 Tonemap 读
        blurBarrier.image = blurImage_[1];
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &blurBarrier);

        // ════════════════════════════════════════════════════════════════════
        // Pass 3: Tone Mapping + Composite → 交换链
        // ════════════════════════════════════════════════════════════════════
        {
            VkClearValue clear{};
            clear.color.float32[3] = 1.0f;
            VkRenderPassBeginInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp.renderPass = tonemapRenderPass_;
            rp.framebuffer = tonemapFramebuffers_[imageIndex];
            rp.renderArea = {{0, 0}, swapchainExtent_};
            rp.clearValueCount = 1;
            rp.pClearValues = &clear;
            vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemapPipeline_);
            VkViewport vp{0, 0, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0, 1};
            vkCmdSetViewport(cmd, 0, 1, &vp);
            VkRect2D sc{{0, 0}, swapchainExtent_};
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemapPipeLayout_, 0, 1, &tonemapDescSet_, 0, nullptr);
            ToneMapPushConstants pc{1.5f, 0.8f, 2.2f}; // exposure, bloomStrength, gamma
            vkCmdPushConstants(cmd, tonemapPipeLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
            vkCmdDraw(cmd, 3, 1, 0, 0); // 全屏三角形
            vkCmdEndRenderPass(cmd);
        }

        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t imgIdx = 0;
        VkResult r = vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX, imageAvailableSems_[currentFrame_], VK_NULL_HANDLE, &imgIdx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }
        updateSceneUBO(currentFrame_);
        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordCommandBuffer(commandBuffers_[currentFrame_], imgIdx);
        VkSemaphore ws[] = {imageAvailableSems_[currentFrame_]};
        VkPipelineStageFlags wst[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore ss[] = {renderFinishedSems_[currentFrame_]};
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = ws;
        si.pWaitDstStageMask = wst;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &commandBuffers_[currentFrame_];
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = ss;
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &si, inFlightFences_[currentFrame_]));
        VkSwapchainKHR scs[] = {swapchain_};
        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = ss;
        pi.swapchainCount = 1;
        pi.pSwapchains = scs;
        pi.pImageIndices = &imgIdx;
        r = vkQueuePresentKHR(presentQueue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        }
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void mainLoop() {
        std::cout << "✨ HDR + Bloom + Tone Mapping 运行中（ESC 退出）\n";
        std::cout << "   观察：高亮物体周围的 Bloom 泛光效果\n";
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    // ─── 辅助函数 ─────────────────────────────────────────────────────────────

    uint32_t findMemoryType(uint32_t f, VkMemoryPropertyFlags p) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((f & (1u << i)) && (mp.memoryTypes[i].propertyFlags & p) == p)
                return i;
        throw std::runtime_error("找不到内存类型");
    }
    void createBuffer(VkDeviceSize sz, VkBufferUsageFlags u, VkMemoryPropertyFlags p, VkBuffer& b, VkDeviceMemory& m) {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = sz;
        ci.usage = u;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &ci, nullptr, &b));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(device_, b, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, p);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &m));
        VK_CHECK(vkBindBufferMemory(device_, b, m, 0));
    }
    VkFormat findSupportedFormat(const std::vector<VkFormat>& c, VkImageTiling t, VkFormatFeatureFlags f) {
        for (VkFormat fmt : c) {
            VkFormatProperties p;
            vkGetPhysicalDeviceFormatProperties(physicalDevice_, fmt, &p);
            if (t == VK_IMAGE_TILING_OPTIMAL && (p.optimalTilingFeatures & f) == f)
                return fmt;
        }
        throw std::runtime_error("找不到格式");
    }
    VkFormat findDepthFormat() {
        return findSupportedFormat({VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                                   VK_IMAGE_TILING_OPTIMAL,
                                   VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }
    void createVertexBuffer() {
        VkDeviceSize sz = sizeof(SCENE[0]) * SCENE.size();
        createBuffer(sz,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_,
                     vertexMemory_);
        void* d = nullptr;
        vkMapMemory(device_, vertexMemory_, 0, sz, 0, &d);
        std::memcpy(d, SCENE.data(), (size_t)sz);
        vkUnmapMemory(device_, vertexMemory_);
    }
    void createInstance() {
        VkApplicationInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.apiVersion = VK_API_VERSION_1_3;
        auto exts = getRequiredInstanceExtensions();
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &ai;
        ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
        ci.ppEnabledExtensionNames = exts.data();
#ifdef __APPLE__
#ifdef __APPLE__
        ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

#endif
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    }
    void createSurface() {
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
    }
    void pickPhysicalDevice() {
        uint32_t c = 0;
        vkEnumeratePhysicalDevices(instance_, &c, nullptr);
        std::vector<VkPhysicalDevice> devs(c);
        vkEnumeratePhysicalDevices(instance_, &c, devs.data());
        for (auto& d : devs)
            if (findQueueFamilies(d, surface_).isComplete() && checkDeviceExtensionSupport(d)) {
                physicalDevice_ = d;
                break;
            }
        if (!physicalDevice_)
            throw std::runtime_error("无合适GPU");
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(physicalDevice_, &p);
        std::cout << "✅ GPU: " << p.deviceName << "\n";
    }
    void createLogicalDevice() {
        queueIndices_ = findQueueFamilies(physicalDevice_, surface_);
        std::set<uint32_t> fams = {queueIndices_.graphicsFamily.value(), queueIndices_.presentFamily.value()};
        const float pri = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t f : fams) {
            VkDeviceQueueCreateInfo q{};
            q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueFamilyIndex = f;
            q.queueCount = 1;
            q.pQueuePriorities = &pri;
            qcis.push_back(q);
        }
        VkPhysicalDeviceFeatures feat{};
        feat.samplerAnisotropy = VK_TRUE;
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos = qcis.data();
        ci.pEnabledFeatures = &feat;
        ci.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_));
        vkGetDeviceQueue(device_, queueIndices_.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueIndices_.presentFamily.value(), 0, &presentQueue_);
    }
    void createSwapchain() {
        auto sc = querySwapChainSupport(physicalDevice_, surface_);
        auto fmt = chooseSwapSurfaceFormat(sc.formats);
        auto mode = chooseSwapPresentMode(sc.presentModes);
        swapchainExtent_ = chooseSwapExtent(sc.capabilities, window_);
        uint32_t n = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            n = std::min(n, sc.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_;
        ci.minImageCount = n;
        ci.imageFormat = fmt.format;
        ci.imageColorSpace = fmt.colorSpace;
        ci.imageExtent = swapchainExtent_;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = sc.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = mode;
        ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
        swapchainImages_.resize(n);
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, swapchainImages_.data());
        swapchainImageFormat_ = fmt.format;
    }
    void createImageViews() {
        swapchainImageViews_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image = swapchainImages_[i];
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = swapchainImageFormat_;
            ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY};
            ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]));
        }
    }
    void createCommandPool() {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
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
        VkSemaphoreCreateInfo sCI{};
        sCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fCI{};
        fCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &sCI, nullptr, &imageAvailableSems_[i]));
            VK_CHECK(vkCreateSemaphore(device_, &sCI, nullptr, &renderFinishedSems_[i]));
            VK_CHECK(vkCreateFence(device_, &fCI, nullptr, &inFlightFences_[i]));
        }
    }
    void recreateSwapchain() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (!w || !h) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        for (auto& fb : tonemapFramebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        createTonemapFramebuffers();
    }
    void cleanup() {
        vkDestroySampler(device_, hdrSampler_, nullptr);
        auto destroyImage = [this](VkImage& i, VkDeviceMemory& m, VkImageView& v) {
            vkDestroyImageView(device_, v, nullptr);
            vkDestroyImage(device_, i, nullptr);
            vkFreeMemory(device_, m, nullptr);
        };
        destroyImage(hdrImage_, hdrMemory_, hdrView_);
        destroyImage(brightImage_, brightMemory_, brightView_);
        destroyImage(depthImage_, depthMemory_, depthView_);
        for (int i = 0; i < 2; ++i)
            destroyImage(blurImage_[i], blurMemory_[i], blurView_[i]);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, sceneUBOs_[i], nullptr);
            vkFreeMemory(device_, sceneUBOMemories_[i], nullptr);
        }
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMemory_, nullptr);
        vkDestroyDescriptorPool(device_, sceneDescPool_, nullptr);
        vkDestroyDescriptorPool(device_, blurDescPool_, nullptr);
        vkDestroyDescriptorPool(device_, tonemapDescPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, sceneSetLayout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, blurSetLayout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, tonemapSetLayout_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        vkDestroyFramebuffer(device_, sceneFramebuffer_, nullptr);
        for (auto& fb : tonemapFramebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyPipeline(device_, scenePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, scenePipeLayout_, nullptr);
        vkDestroyPipeline(device_, blurPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, blurPipeLayout_, nullptr);
        vkDestroyPipeline(device_, tonemapPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, tonemapPipeLayout_, nullptr);
        vkDestroyRenderPass(device_, sceneRenderPass_, nullptr);
        vkDestroyRenderPass(device_, tonemapRenderPass_, nullptr);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
        std::cout << "✅ 清理完成。\n";
    }
};

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════"
                 "════\n";
    std::cout << " 第24章：后处理（HDR + Bloom + Tone Mapping + Gamma）\n";
    std::cout << "\n";
    std::cout << " 三趟渲染：\n";
    std::cout << "   Pass1: 场景渲染 → RGBA16F HDR + Bright MRT\n";
    std::cout << "   Pass2: Gaussian Blur (Compute) 2次 → Bloom\n";
    std::cout << "   Pass3: HDR+Bloom → ACES Tone Map → Gamma → 屏幕\n";
    std::cout << "═══════════════════════════════════════════════════════════════"
                 "════\n\n";
    Ch24App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
    return 0;
}
