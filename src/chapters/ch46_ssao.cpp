/**
 * @file ch46_ssao.cpp
 * @brief 第46章：屏幕空间环境光遮蔽（SSAO）
 *
 * 管线：G-Buffer（位置+法线+颜色）→ SSAO → 计算模糊 → 光照合成。
 * 若 RGBA16F 不支持则降级为 RGBA8（features.hpp）。
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
#include <random>
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;
constexpr int SSAO_KERNEL = 32;

struct GVertex { glm::vec3 pos; glm::vec3 normal; glm::vec3 color; glm::vec2 uv; };

static const std::vector<GVertex> SCENE = {
    {{-3,-0.5f,-3},{0,1,0},{0.5f,0.5f,0.5f},{0,0}}, {{3,-0.5f,-3},{0,1,0},{0.5f,0.5f,0.5f},{1,0}},
    {{3,-0.5f,3},{0,1,0},{0.5f,0.5f,0.5f},{1,1}}, {{-3,-0.5f,-3},{0,1,0},{0.5f,0.5f,0.5f},{0,0}},
    {{3,-0.5f,3},{0,1,0},{0.5f,0.5f,0.5f},{1,1}}, {{-3,-0.5f,3},{0,1,0},{0.5f,0.5f,0.5f},{0,1}},
    {{-0.5f,-0.48f,-0.5f},{0,0,-1},{0.85f,0.35f,0.35f},{0,0}}, {{0.5f,-0.48f,-0.5f},{0,0,-1},{0.85f,0.35f,0.35f},{1,0}},
    {{0.5f,1.0f,-0.5f},{0,0,-1},{0.85f,0.35f,0.35f},{1,1}}, {{-0.5f,-0.48f,-0.5f},{0,0,-1},{0.85f,0.35f,0.35f},{0,0}},
    {{0.5f,1.0f,-0.5f},{0,0,-1},{0.85f,0.35f,0.35f},{1,1}}, {{-0.5f,1.0f,-0.5f},{0,0,-1},{0.85f,0.35f,0.35f},{0,1}},
};

struct GeomUBO { alignas(16) glm::mat4 model; alignas(16) glm::mat4 view; alignas(16) glm::mat4 projection; };
struct SSAOUBO {
    alignas(16) glm::mat4 projection;
    alignas(16) glm::mat4 invProjection;
    alignas(16) glm::vec4 samples[SSAO_KERNEL];
    alignas(16) glm::vec2 noiseScale;
    float radius;
    float bias;
};
struct LightUBO { alignas(16) glm::vec4 lightDir; alignas(16) glm::vec4 lightColor; alignas(16) glm::vec4 cameraPos; };

class Ch46App {
public:
    void run() { initWindow(); initVulkan(); mainLoop(); cleanup(); }

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
    VkFormat gBufferFormat_ = VK_FORMAT_R16G16B16A16_SFLOAT;
    DepthResources depth_{};

    VkImage posImage_ = VK_NULL_HANDLE; VkDeviceMemory posMem_ = VK_NULL_HANDLE; VkImageView posView_ = VK_NULL_HANDLE;
    VkImage nrmImage_ = VK_NULL_HANDLE; VkDeviceMemory nrmMem_ = VK_NULL_HANDLE; VkImageView nrmView_ = VK_NULL_HANDLE;
    VkImage albImage_ = VK_NULL_HANDLE; VkDeviceMemory albMem_ = VK_NULL_HANDLE; VkImageView albView_ = VK_NULL_HANDLE;
    VkImage aoRawImage_ = VK_NULL_HANDLE; VkDeviceMemory aoRawMem_ = VK_NULL_HANDLE; VkImageView aoRawView_ = VK_NULL_HANDLE;
    VkImage aoBlurImage_ = VK_NULL_HANDLE; VkDeviceMemory aoBlurMem_ = VK_NULL_HANDLE; VkImageView aoBlurView_ = VK_NULL_HANDLE;
    TextureImage noiseTex_{};

    VkRenderPass gBufferRP_ = VK_NULL_HANDLE;
    VkRenderPass ssaoRP_ = VK_NULL_HANDLE;
    VkRenderPass compositeRP_ = VK_NULL_HANDLE;
    VkFramebuffer gBufferFB_ = VK_NULL_HANDLE;
    VkFramebuffer ssaoFB_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> compositeFBs_;

    VkPipeline geomPipeline_ = VK_NULL_HANDLE;
    VkPipeline ssaoPipeline_ = VK_NULL_HANDLE;
    VkPipeline compositePipeline_ = VK_NULL_HANDLE;
    VkPipeline blurPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout geomLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout ssaoLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout compositeLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout blurLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout geomDSL_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssaoDSL_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout compositeDSL_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout blurDSL_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> geomSets_;
    std::vector<VkDescriptorSet> ssaoSets_;
    std::vector<VkDescriptorSet> compositeSets_;
    VkDescriptorSet blurSet_ = VK_NULL_HANDLE;

    VkBuffer vertexBuffer_ = VK_NULL_HANDLE; VkDeviceMemory vertexMem_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> geomUBOs_; std::vector<VkDeviceMemory> geomUBOMem_; std::vector<void*> geomMapped_;
    std::vector<VkBuffer> ssaoUBOs_; std::vector<VkDeviceMemory> ssaoUBOMem_; std::vector<void*> ssaoMapped_;
    std::vector<VkBuffer> lightUBOs_; std::vector<VkDeviceMemory> lightUBOMem_; std::vector<void*> lightMapped_;
    VkSampler linearSampler_ = VK_NULL_HANDLE;

    std::vector<VkImage> swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<VkCommandBuffer> cmdBuffers_;
    std::vector<VkSemaphore> imageAvailable_;
    std::vector<VkSemaphore> renderFinished_;
    std::vector<VkFence> inFlight_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;
    InteractiveChapterTools interactive_;

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch46 - SSAO（G-Buffer + 模糊）", nullptr, nullptr);
        interactive_.attachInput(window_);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch46App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan()
    {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physicalDevice_);
        createLogicalDevice(physicalDevice_, surface_, device_, graphicsQueue_, presentQueue_, queueIndices_);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physicalDevice_, device_, extent_);
        selectGBufferFormat();
        createCommandPool();
        createOffscreenImages();
        createNoiseTexture();
        createSampler();
        createRenderPasses();
        createDescriptorLayouts();
        createPipelines();
        createFramebuffers();
        createVertexBuffer();
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
        ii.renderPass = compositeRP_;
        ii.swapchainFormat = swapFormat_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setTarget(glm::vec3(0.0f));
        interactive_.camera().setDistance(6.0f);
        std::cout << "✅ SSAO 初始化完成（G-Buffer 格式: "
                  << (gBufferFormat_ == VK_FORMAT_R16G16B16A16_SFLOAT ? "RGBA16F" : "RGBA8 降级") << "）\n";
    }

    void selectGBufferFormat()
    {
        const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if (!isFormatSupported(physicalDevice_, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, needed)) {
            logFeatureFallback("RGBA16F G-Buffer", "RGBA8 UNORM");
            gBufferFormat_ = VK_FORMAT_R8G8B8A8_UNORM;
        }
    }

    void createColorImage(VkImage& img, VkDeviceMemory& mem, VkImageView& view, VkImageUsageFlags usage)
    {
        createImage(physicalDevice_, device_, extent_.width, extent_.height, gBufferFormat_,
                    VK_IMAGE_TILING_OPTIMAL, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, img, mem);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = gBufferFormat_;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &view));
    }

    void createOffscreenImages()
    {
        const VkImageUsageFlags gUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        createColorImage(posImage_, posMem_, posView_, gUsage);
        createColorImage(nrmImage_, nrmMem_, nrmView_, gUsage);
        createColorImage(albImage_, albMem_, albView_, gUsage);
        createImage(physicalDevice_, device_, extent_.width, extent_.height, VK_FORMAT_R8_UNORM,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, aoRawImage_, aoRawMem_);
        VkImageViewCreateInfo aoView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        aoView.image = aoRawImage_;
        aoView.viewType = VK_IMAGE_VIEW_TYPE_2D;
        aoView.format = VK_FORMAT_R8_UNORM;
        aoView.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &aoView, nullptr, &aoRawView_));
        createImage(physicalDevice_, device_, extent_.width, extent_.height, VK_FORMAT_R8_UNORM,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, aoBlurImage_, aoBlurMem_);
        aoView.image = aoBlurImage_;
        VK_CHECK(vkCreateImageView(device_, &aoView, nullptr, &aoBlurView_));
    }

    void createNoiseTexture()
    {
        ImageData noise;
        noise.width = 4;
        noise.height = 4;
        noise.channels = 4;
        noise.pixels.resize(4 * 4 * 4);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::default_random_engine rng(42);
        for (size_t i = 0; i < 16; ++i) {
            glm::vec3 v(dist(rng), dist(rng), 0.0f);
            v = glm::normalize(v);
            noise.pixels[i * 4 + 0] = static_cast<uint8_t>((v.x * 0.5f + 0.5f) * 255.0f);
            noise.pixels[i * 4 + 1] = static_cast<uint8_t>((v.y * 0.5f + 0.5f) * 255.0f);
            noise.pixels[i * 4 + 2] = 127;
            noise.pixels[i * 4 + 3] = 255;
        }
        noiseTex_ = createTextureFromImageData(physicalDevice_, device_, commandPool_, graphicsQueue_, noise, false);
    }

    void createSampler()
    {
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter = ci.minFilter = VK_FILTER_LINEAR;
        ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &linearSampler_));
    }

    VkAttachmentDescription colorAttachment(VkFormat fmt, VkImageLayout finalLayout) const
    {
        VkAttachmentDescription att{};
        att.format = fmt;
        att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout = finalLayout;
        return att;
    }

    void createRenderPasses()
    {
        std::array<VkAttachmentDescription, 4> gAtt = {
            colorAttachment(gBufferFormat_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            colorAttachment(gBufferFormat_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            colorAttachment(gBufferFormat_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            colorAttachment(depth_.format, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)};
        gAtt[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        std::array<VkAttachmentReference, 3> colorRefs = {{{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}}};
        VkAttachmentReference depthRef{3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 3;
        sub.pColorAttachments = colorRefs.data();
        sub.pDepthStencilAttachment = &depthRef;
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.attachmentCount = static_cast<uint32_t>(gAtt.size());
        rp.pAttachments = gAtt.data();
        rp.subpassCount = 1;
        rp.pSubpasses = &sub;
        VK_CHECK(vkCreateRenderPass(device_, &rp, nullptr, &gBufferRP_));

        VkAttachmentDescription aoAtt = colorAttachment(VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VkAttachmentReference aoRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription aoSub{};
        aoSub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        aoSub.colorAttachmentCount = 1;
        aoSub.pColorAttachments = &aoRef;
        rp.attachmentCount = 1;
        rp.pAttachments = &aoAtt;
        rp.pSubpasses = &aoSub;
        VK_CHECK(vkCreateRenderPass(device_, &rp, nullptr, &ssaoRP_));

        VkAttachmentDescription outAtt = colorAttachment(swapFormat_, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        VkAttachmentReference outRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription compSub{};
        compSub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        compSub.colorAttachmentCount = 1;
        compSub.pColorAttachments = &outRef;
        rp.pAttachments = &outAtt;
        rp.pSubpasses = &compSub;
        VK_CHECK(vkCreateRenderPass(device_, &rp, nullptr, &compositeRP_));
    }

    VkShaderModule loadShader(const char* name) { return createShaderModuleFromFile(device_, name); }

    VkPipeline buildFullscreenFragPipeline(VkRenderPass rp, VkPipelineLayout layout, const char* fragName)
    {
        VkShaderModule vert = loadShader("deferred_lighting.vert.spv");
        VkShaderModule frag = loadShader(fragName);
        VkPipelineShaderStageCreateInfo stages[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr}};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
                                                  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0,
                                                VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 1.0f, nullptr, VK_FALSE, VK_FALSE};
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dyn};
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.stageCount = 2;
        gp.pStages = stages;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vps;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pColorBlendState = &cb;
        gp.pDynamicState = &dynS;
        gp.layout = layout;
        gp.renderPass = rp;
        VkPipeline pipe = VK_NULL_HANDLE;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        return pipe;
    }

    void createDescriptorLayouts()
    {
        VkDescriptorSetLayoutBinding gBind{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 1;
        ci.pBindings = &gBind;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &geomDSL_));
        std::array<VkDescriptorSetLayoutBinding, 4> sBinds = {{
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
        ci.bindingCount = static_cast<uint32_t>(sBinds.size());
        ci.pBindings = sBinds.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &ssaoDSL_));
        std::array<VkDescriptorSetLayoutBinding, 5> cBinds = {{
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
        ci.bindingCount = static_cast<uint32_t>(cBinds.size());
        ci.pBindings = cBinds.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &compositeDSL_));
        std::array<VkDescriptorSetLayoutBinding, 2> bBinds = {{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}}};
        ci.bindingCount = static_cast<uint32_t>(bBinds.size());
        ci.pBindings = bBinds.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &blurDSL_));
    }

    void createPipelines()
    {
        VkShaderModule gVert = loadShader("gbuffer.vert.spv");
        VkShaderModule gFrag = loadShader("gbuffer.frag.spv");
        VkPipelineShaderStageCreateInfo gStages[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, gVert, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, gFrag, "main", nullptr}};
        VkVertexInputBindingDescription bind{0, sizeof(GVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 4> attrs{};
        attrs[0].location = 0; attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = offsetof(GVertex, pos);
        attrs[1].location = 1; attrs[1].binding = 0;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = offsetof(GVertex, normal);
        attrs[2].location = 2; attrs[2].binding = 0;
        attrs[2].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[2].offset = offsetof(GVertex, color);
        attrs[3].location = 3; attrs[3].binding = 0;
        attrs[3].format = VK_FORMAT_R32G32_SFLOAT; attrs[3].offset = offsetof(GVertex, uv);
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
        vi.pVertexAttributeDescriptions = attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
                                                  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0,
                                                VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 1.0f, nullptr, VK_FALSE, VK_FALSE};
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;
        std::array<VkPipelineColorBlendAttachmentState, 3> cbas{};
        for (auto& c : cbas) c.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 3;
        cb.pAttachments = cbas.data();
        VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dyn};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &geomDSL_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pl, nullptr, &geomLayout_));
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.stageCount = 2;
        gp.pStages = gStages;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vps;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cb;
        gp.pDynamicState = &dynS;
        gp.layout = geomLayout_;
        gp.renderPass = gBufferRP_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &geomPipeline_));
        vkDestroyShaderModule(device_, gFrag, nullptr);
        vkDestroyShaderModule(device_, gVert, nullptr);

        pl.pSetLayouts = &ssaoDSL_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pl, nullptr, &ssaoLayout_));
        ssaoPipeline_ = buildFullscreenFragPipeline(ssaoRP_, ssaoLayout_, "ssao.frag.spv");

        pl.pSetLayouts = &compositeDSL_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pl, nullptr, &compositeLayout_));
        compositePipeline_ = buildFullscreenFragPipeline(compositeRP_, compositeLayout_, "ssao_composite.frag.spv");

        pl.pSetLayouts = &blurDSL_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pl, nullptr, &blurLayout_));
        VkShaderModule blur = loadShader("ssao_blur.comp.spv");
        VkPipelineShaderStageCreateInfo cs{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                           VK_SHADER_STAGE_COMPUTE_BIT, blur, "main", nullptr};
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage = cs;
        cp.layout = blurLayout_;
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cp, nullptr, &blurPipeline_));
        vkDestroyShaderModule(device_, blur, nullptr);
    }

    void createFramebuffers()
    {
        std::array<VkImageView, 4> gAtt = {posView_, nrmView_, albView_, depth_.view};
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = gBufferRP_;
        fb.attachmentCount = static_cast<uint32_t>(gAtt.size());
        fb.pAttachments = gAtt.data();
        fb.width = extent_.width;
        fb.height = extent_.height;
        fb.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &fb, nullptr, &gBufferFB_));
        VkImageView aoAtt = aoRawView_;
        fb.renderPass = ssaoRP_;
        fb.attachmentCount = 1;
        fb.pAttachments = &aoAtt;
        VK_CHECK(vkCreateFramebuffer(device_, &fb, nullptr, &ssaoFB_));
        compositeFBs_.resize(swapViews_.size());
        for (size_t i = 0; i < swapViews_.size(); ++i) {
            fb.renderPass = compositeRP_;
            fb.pAttachments = &swapViews_[i];
            VK_CHECK(vkCreateFramebuffer(device_, &fb, nullptr, &compositeFBs_[i]));
        }
    }

    void updateUniforms(uint32_t frame)
    {
        GeomUBO g{};
        g.model = glm::mat4(1.0f);
        const float aspect = extent_.width / static_cast<float>(extent_.height);
        g.view = interactive_.camera().viewMatrix();
        g.projection = interactive_.camera().projectionMatrix(aspect, 60.0f, 0.1f, 30.0f);
        std::memcpy(geomMapped_[frame], &g, sizeof(g));
        SSAOUBO s{};
        s.projection = g.projection;
        s.invProjection = glm::inverse(g.projection);
        s.noiseScale = glm::vec2(extent_.width / 4.0f, extent_.height / 4.0f);
        s.radius = 0.45f;
        s.bias = 0.025f;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::default_random_engine rng(7);
        for (int i = 0; i < SSAO_KERNEL; ++i) {
            glm::vec3 sample(dist(rng) * 2.0f - 1.0f, dist(rng) * 2.0f - 1.0f, dist(rng));
            sample = glm::normalize(sample);
            sample *= dist(rng);
            const float scale = static_cast<float>(i) / static_cast<float>(SSAO_KERNEL);
            sample *= glm::mix(0.1f, 1.0f, scale * scale);
            s.samples[i] = glm::vec4(sample, 0.0f);
        }
        std::memcpy(ssaoMapped_[frame], &s, sizeof(s));
        LightUBO l{};
        l.lightDir = glm::vec4(glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f)), 0.0f);
        l.lightColor = glm::vec4(1.0f, 0.95f, 0.85f, 1.0f);
        l.cameraPos = glm::vec4(interactive_.camera().eyePosition(), 1.0f);
        std::memcpy(lightMapped_[frame], &l, sizeof(l));
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginInfo);
        interactive_.beginGpuSection(cmd, currentFrame_);
        std::array<VkClearValue, 4> gClears{};
        gClears[3].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo gRp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        gRp.renderPass = gBufferRP_;
        gRp.framebuffer = gBufferFB_;
        gRp.renderArea = {{0, 0}, extent_};
        gRp.clearValueCount = static_cast<uint32_t>(gClears.size());
        gRp.pClearValues = gClears.data();
        vkCmdBeginRenderPass(cmd, &gRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geomPipeline_);
        VkViewport vp{0, 0, (float)extent_.width, (float)extent_.height, 0, 1};
        VkRect2D sc{{0, 0}, extent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        VkBuffer vb[] = {vertexBuffer_};
        VkDeviceSize off[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geomLayout_, 0, 1, &geomSets_[currentFrame_], 0, nullptr);
        vkCmdDraw(cmd, static_cast<uint32_t>(SCENE.size()), 1, 0, 0);
        vkCmdEndRenderPass(cmd);

        VkClearValue aoClear{};
        aoClear.color = {{1.0f, 0, 0, 0}};
        VkRenderPassBeginInfo aoRp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        aoRp.renderPass = ssaoRP_;
        aoRp.framebuffer = ssaoFB_;
        aoRp.renderArea = {{0, 0}, extent_};
        aoRp.clearValueCount = 1;
        aoRp.pClearValues = &aoClear;
        vkCmdBeginRenderPass(cmd, &aoRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssaoPipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssaoLayout_, 0, 1, &ssaoSets_[currentFrame_], 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);

        VkImageMemoryBarrier aoToCompute{};
        aoToCompute.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        aoToCompute.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        aoToCompute.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        aoToCompute.srcQueueFamilyIndex = aoToCompute.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        aoToCompute.image = aoRawImage_;
        aoToCompute.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        aoToCompute.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        aoToCompute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &aoToCompute);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, blurPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, blurLayout_, 0, 1, &blurSet_, 0, nullptr);
        vkCmdDispatch(cmd, (extent_.width + 15) / 16, (extent_.height + 15) / 16, 1);

        VkImageMemoryBarrier aoToSample = aoToCompute;
        aoToSample.image = aoBlurImage_;
        aoToSample.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        aoToSample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        aoToSample.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        aoToSample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &aoToSample);

        VkClearValue compClear{};
        compClear.color = {{0.02f, 0.02f, 0.05f, 1.0f}};
        VkRenderPassBeginInfo cRp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        cRp.renderPass = compositeRP_;
        cRp.framebuffer = compositeFBs_[imageIndex];
        cRp.renderArea = {{0, 0}, extent_};
        cRp.clearValueCount = 1;
        cRp.pClearValues = &compClear;
        vkCmdBeginRenderPass(cmd, &cRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositeLayout_, 0, 1, &compositeSets_[currentFrame_], 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);
        interactive_.endGpuSection(cmd, currentFrame_);
        vkEndCommandBuffer(cmd);
    }

    void drawFrame()
    {
        vkWaitForFences(device_, 1, &inFlight_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t img = 0;
        if (vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailable_[currentFrame_], VK_NULL_HANDLE, &img) == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }
        updateUniforms(currentFrame_);
        vkResetFences(device_, 1, &inFlight_[currentFrame_]);
        vkResetCommandBuffer(cmdBuffers_[currentFrame_], 0);
        recordCommandBuffer(cmdBuffers_[currentFrame_], img);
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &imageAvailable_[currentFrame_];
        submit.pWaitDstStageMask = &stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmdBuffers_[currentFrame_];
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
        if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || resized_) { resized_ = false; recreateSwapchain(); }
        interactive_.endFrame(currentFrame_);
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void mainLoop()
    {
        std::cout << "🌫️  SSAO 运行中（ESC 退出）\n";
        auto lastTime = std::chrono::steady_clock::now();
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            const auto now = std::chrono::steady_clock::now();
            const float dt = std::chrono::duration<float>(now - lastTime).count();
            lastTime = now;
            interactive_.beginFrame(dt);
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    void createSwapchain()
    {
        auto sc = querySwapChainSupport(physicalDevice_, surface_);
        auto fmt = chooseSwapSurfaceFormat(sc.formats);
        extent_ = chooseSwapExtent(sc.capabilities, window_);
        uint32_t count = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0) count = std::min(count, sc.capabilities.maxImageCount);
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
        ci.presentMode = chooseSwapPresentMode(sc.presentModes);
        ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        vkGetSwapchainImagesKHR(device_, swapchain_, &count, nullptr);
        swapImages_.resize(count);
        vkGetSwapchainImagesKHR(device_, swapchain_, &count, swapImages_.data());
        swapFormat_ = fmt.format;
    }

    void createImageViews()
    {
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

    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
    }

    void createVertexBuffer()
    {
        const VkDeviceSize sz = sizeof(GVertex) * SCENE.size();
        createBuffer(physicalDevice_, device_, sz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vertexBuffer_, vertexMem_);
        void* mapped = nullptr;
        vkMapMemory(device_, vertexMem_, 0, sz, 0, &mapped);
        std::memcpy(mapped, SCENE.data(), static_cast<size_t>(sz));
        vkUnmapMemory(device_, vertexMem_);
    }

    void createUniformBuffers()
    {
        geomUBOs_.resize(MAX_FRAMES); geomUBOMem_.resize(MAX_FRAMES); geomMapped_.resize(MAX_FRAMES);
        ssaoUBOs_.resize(MAX_FRAMES); ssaoUBOMem_.resize(MAX_FRAMES); ssaoMapped_.resize(MAX_FRAMES);
        lightUBOs_.resize(MAX_FRAMES); lightUBOMem_.resize(MAX_FRAMES); lightMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physicalDevice_, device_, sizeof(GeomUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, geomUBOs_[i], geomUBOMem_[i]);
            vkMapMemory(device_, geomUBOMem_[i], 0, sizeof(GeomUBO), 0, &geomMapped_[i]);
            createBuffer(physicalDevice_, device_, sizeof(SSAOUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ssaoUBOs_[i], ssaoUBOMem_[i]);
            vkMapMemory(device_, ssaoUBOMem_[i], 0, sizeof(SSAOUBO), 0, &ssaoMapped_[i]);
            createBuffer(physicalDevice_, device_, sizeof(LightUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, lightUBOs_[i], lightUBOMem_[i]);
            vkMapMemory(device_, lightUBOMem_[i], 0, sizeof(LightUBO), 0, &lightMapped_[i]);
        }
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 3> sizes = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES * 3},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES * 8 + 2},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}}};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        ci.pPoolSizes = sizes.data();
        ci.maxSets = MAX_FRAMES * 3 + 1;
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool_));
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> gLayouts(MAX_FRAMES, geomDSL_);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool_;
        ai.descriptorSetCount = MAX_FRAMES;
        ai.pSetLayouts = gLayouts.data();
        geomSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, geomSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{geomUBOs_[i], 0, sizeof(GeomUBO)};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, geomSets_[i], 0, 0, 1,
                                   VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bi, nullptr};
            vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
        }
        std::vector<VkDescriptorSetLayout> sLayouts(MAX_FRAMES, ssaoDSL_);
        ai.pSetLayouts = sLayouts.data();
        ssaoSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, ssaoSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorImageInfo pos{linearSampler_, posView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo nrm{linearSampler_, nrmView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo noise{noiseTex_.sampler, noiseTex_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorBufferInfo sbi{ssaoUBOs_[i], 0, sizeof(SSAOUBO)};
            std::array<VkWriteDescriptorSet, 4> ws = {{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ssaoSets_[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &pos, nullptr, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ssaoSets_[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &nrm, nullptr, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ssaoSets_[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &noise, nullptr, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ssaoSets_[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sbi, nullptr}}};
            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(ws.size()), ws.data(), 0, nullptr);
        }
        std::vector<VkDescriptorSetLayout> cLayouts(MAX_FRAMES, compositeDSL_);
        ai.pSetLayouts = cLayouts.data();
        compositeSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, compositeSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorImageInfo alb{linearSampler_, albView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo nrm{linearSampler_, nrmView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo pos{linearSampler_, posView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo ao{linearSampler_, aoBlurView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorBufferInfo lbi{lightUBOs_[i], 0, sizeof(LightUBO)};
            std::array<VkWriteDescriptorSet, 5> ws = {{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compositeSets_[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &alb, nullptr, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compositeSets_[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &nrm, nullptr, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compositeSets_[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &pos, nullptr, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compositeSets_[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ao, nullptr, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compositeSets_[i], 4, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &lbi, nullptr}}};
            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(ws.size()), ws.data(), 0, nullptr);
        }
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &blurDSL_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &blurSet_));
        VkDescriptorImageInfo inImg{VK_NULL_HANDLE, aoRawView_, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo outImg{VK_NULL_HANDLE, aoBlurView_, VK_IMAGE_LAYOUT_GENERAL};
        std::array<VkWriteDescriptorSet, 2> bws = {{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, blurSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &inImg, nullptr, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, blurSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImg, nullptr, nullptr}}};
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(bws.size()), bws.data(), 0, nullptr);
    }

    void createCommandBuffers()
    {
        cmdBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = MAX_FRAMES;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, cmdBuffers_.data()));
    }

    void createSyncObjects()
    {
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

    void destroyOffscreen()
    {
        auto destroyColor = [&](VkImage& i, VkDeviceMemory& m, VkImageView& v) {
            vkDestroyImageView(device_, v, nullptr);
            vkDestroyImage(device_, i, nullptr);
            vkFreeMemory(device_, m, nullptr);
        };
        destroyColor(posImage_, posMem_, posView_);
        destroyColor(nrmImage_, nrmMem_, nrmView_);
        destroyColor(albImage_, albMem_, albView_);
        destroyColor(aoRawImage_, aoRawMem_, aoRawView_);
        destroyColor(aoBlurImage_, aoBlurMem_, aoBlurView_);
    }

    void recreateSwapchain()
    {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) { glfwGetFramebufferSize(window_, &w, &h); glfwWaitEvents(); }
        vkDeviceWaitIdle(device_);
        vkDestroyFramebuffer(device_, gBufferFB_, nullptr);
        vkDestroyFramebuffer(device_, ssaoFB_, nullptr);
        for (auto fb : compositeFBs_) vkDestroyFramebuffer(device_, fb, nullptr);
        destroyOffscreen();
        destroyDepthResources(device_, depth_);
        for (auto v : swapViews_) vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physicalDevice_, device_, extent_);
        createOffscreenImages();
        createFramebuffers();
        VK_CHECK(vkResetDescriptorPool(device_, pool_, 0));
        createDescriptorSets();
        interactive_.onSwapchainRecreated(compositeRP_, swapFormat_,
            static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup()
    {
        destroyTexture(device_, noiseTex_);
        vkDestroySampler(device_, linearSampler_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, geomUBOs_[i], nullptr);
            vkFreeMemory(device_, geomUBOMem_[i], nullptr);
            vkDestroyBuffer(device_, ssaoUBOs_[i], nullptr);
            vkFreeMemory(device_, ssaoUBOMem_[i], nullptr);
            vkDestroyBuffer(device_, lightUBOs_[i], nullptr);
            vkFreeMemory(device_, lightUBOMem_[i], nullptr);
        }
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, geomDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, ssaoDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, compositeDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, blurDSL_, nullptr);
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMem_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailable_[i], nullptr);
            vkDestroySemaphore(device_, renderFinished_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        vkDestroyFramebuffer(device_, gBufferFB_, nullptr);
        vkDestroyFramebuffer(device_, ssaoFB_, nullptr);
        for (auto fb : compositeFBs_) vkDestroyFramebuffer(device_, fb, nullptr);
        destroyOffscreen();
        destroyDepthResources(device_, depth_);
        vkDestroyPipeline(device_, geomPipeline_, nullptr);
        vkDestroyPipeline(device_, ssaoPipeline_, nullptr);
        vkDestroyPipeline(device_, compositePipeline_, nullptr);
        vkDestroyPipeline(device_, blurPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, geomLayout_, nullptr);
        vkDestroyPipelineLayout(device_, ssaoLayout_, nullptr);
        vkDestroyPipelineLayout(device_, compositeLayout_, nullptr);
        vkDestroyPipelineLayout(device_, blurLayout_, nullptr);
        vkDestroyRenderPass(device_, gBufferRP_, nullptr);
        vkDestroyRenderPass(device_, ssaoRP_, nullptr);
        vkDestroyRenderPass(device_, compositeRP_, nullptr);
        for (auto v : swapViews_) vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
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
    std::cout << "══════════════════════════════════════════════════\n";
    std::cout << " 第46章：SSAO（G-Buffer → 遮蔽 → 模糊 → 合成）\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    Ch46App app;
    try { app.run(); } catch (const std::exception& e) { std::cerr << "❌ " << e.what() << "\n"; return 1; }
    return 0;
}
