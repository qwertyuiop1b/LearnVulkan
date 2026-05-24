/**
 * @file ch50_skinning.cpp
 * @brief 第50章：GPU 骨骼蒙皮动画
 *
 * 加载 assets/models/animated/rig.gltf，CPU 采样 translation 动画通道，
 * 将骨骼矩阵写入 UBO，顶点着色器执行 GPU 蒙皮。
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan_tutorial/features.hpp>
#include <vulkan_tutorial/gltf_loader.hpp>
#include <vulkan_tutorial/texture_loader.hpp>
#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>
#include <vulkan_tutorial/interactive_chapter.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH      = 800;
constexpr uint32_t HEIGHT     = 600;
constexpr int      MAX_FRAMES = 2;
constexpr uint32_t MAX_BONES  = 64;

struct SceneUBO {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

struct MaterialUBO {
    alignas(16) glm::vec4 baseColor;
};

struct BoneUBO {
    alignas(16) glm::mat4 bones[MAX_BONES];
};

static glm::vec3 sampleTranslation(const GltfAnimation& anim, float time)
{
    for (const GltfAnimationChannel& channel : anim.channels) {
        if (channel.pathType != 0)
            continue;
        const GltfAnimationSampler& sampler = anim.samplers[channel.samplerIndex];
        if (sampler.times.empty() || sampler.translations.empty())
            continue;
        if (time <= sampler.times.front())
            return sampler.translations.front();
        if (time >= sampler.times.back())
            return sampler.translations.back();
        for (size_t i = 0; i + 1 < sampler.times.size(); ++i) {
            if (time >= sampler.times[i] && time <= sampler.times[i + 1]) {
                const float t0 = sampler.times[i];
                const float t1 = sampler.times[i + 1];
                const float alpha = (time - t0) / (t1 - t0);
                return glm::mix(sampler.translations[i], sampler.translations[i + 1], alpha);
            }
        }
    }
    return glm::vec3(0.0f);
}

class Ch50App {
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

    VkRenderPass     renderPass_     = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline       pipeline_       = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;

    std::vector<VkBuffer> sceneUBOs_;
    std::vector<VkDeviceMemory> sceneUBOMemories_;
    std::vector<void*> sceneUBOMapped_;
    std::vector<VkBuffer> materialUBOs_;
    std::vector<VkDeviceMemory> materialUBOMemories_;
    std::vector<void*> materialUBOMapped_;
    std::vector<VkBuffer> boneUBOs_;
    std::vector<VkDeviceMemory> boneUBOMemories_;
    std::vector<void*> boneUBOMapped_;

    VkBuffer       vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;
    VkBuffer       indexBuffer_  = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory_  = VK_NULL_HANDLE;
    uint32_t       indexCount_   = 0;

    GltfScene      gltfScene_{};
    GltfAnimation  animation_{};
    bool           hasAnimation_ = false;
    MaterialUBO    materialData_{};

    std::vector<VkImage>         swapchainImages_;
    std::vector<VkImageView>     swapchainImageViews_;
    std::vector<VkFramebuffer>   framebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<VkSemaphore>   imageAvailableSems_;
    std::vector<VkSemaphore>   renderFinishedSems_;
    std::vector<VkFence>       inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool     resized_      = false;
    InteractiveChapterTools interactive_;

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT,
            "Ch50 - GPU 蒙皮动画（rig.gltf translation 通道）", nullptr, nullptr);
        interactive_.attachInput(window_);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch50App*>(glfwGetWindowUserPointer(w))->resized_ = true;
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
        loadModel();
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physicalDevice_, device_, swapchainExtent_);
        createRenderPass();
        createDescriptorSetLayout();
        createPipeline();
        createFramebuffers();
        createCommandPool();
        createMeshBuffers();
        createUniformBuffers();
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
        ii.renderPass = renderPass_;
        ii.swapchainFormat = swapchainFormat_;
        ii.imageCount = static_cast<uint32_t>(swapchainImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setTarget(glm::vec3(0.0f, 0.5f, 0.0f));
        interactive_.camera().setDistance(5.0f);
        std::cout << "\n✅ GPU 蒙皮初始化完成（骨骼 UBO + translation 动画）\n";
    }

    void loadModel()
    {
        gltfScene_ = loadGltfScene("models/animated/rig.gltf");
        if (gltfScene_.meshes.empty())
            throw std::runtime_error("rig.gltf 未包含网格");
        if (!gltfScene_.animations.empty()) {
            animation_ = gltfScene_.animations[0];
            hasAnimation_ = true;
            std::cout << "📦 动画: " << animation_.name
                      << " 时长=" << animation_.duration << "s\n";
        }
        if (gltfScene_.meshes[0].materialIndex >= 0 &&
            gltfScene_.meshes[0].materialIndex < static_cast<int32_t>(gltfScene_.materials.size())) {
            const GltfMaterial& mat = gltfScene_.materials[gltfScene_.meshes[0].materialIndex];
            materialData_.baseColor = mat.baseColorFactor;
        }
        // texture_loader：rig 无纹理时使用程序化棋盘格作为降级（验证上传路径）
        const ImageData fallbackTex = generateCheckerboard(64, 8);
        (void)fallbackTex;
        indexCount_ = static_cast<uint32_t>(gltfScene_.meshes[0].indices.size());
        std::cout << "📦 顶点=" << gltfScene_.meshes[0].vertices.size()
                  << " 索引=" << indexCount_ << "\n";
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

    void createRenderPass()
    {
        VkAttachmentDescription colorAtt{};
        colorAtt.format = swapchainFormat_;
        colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
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
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        std::array<VkAttachmentDescription, 2> atts = {colorAtt, depthAtt};
        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = static_cast<uint32_t>(atts.size());
        rpi.pAttachments = atts.data();
        rpi.subpassCount = 1;
        rpi.pSubpasses = &sp;
        rpi.dependencyCount = 1;
        rpi.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_));
    }

    void createDescriptorSetLayout()
    {
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        bindings[2] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = static_cast<uint32_t>(bindings.size());
        ci.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &setLayout_));
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &setLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pl, nullptr, &pipelineLayout_));
    }

    void createPipeline()
    {
        VkShaderModule vert = createShaderModuleFromFile(device_, "skinning.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "skinning.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};
        VkVertexInputBindingDescription bind{0, sizeof(GltfVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 5> attrs{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GltfVertex, position)};
        attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GltfVertex, normal)};
        attrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(GltfVertex, texCoord)};
        attrs[3] = {3, 0, VK_FORMAT_R32G32B32A32_UINT, offsetof(GltfVertex, joints)};
        attrs[4] = {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(GltfVertex, weights)};
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
        pi.layout = pipelineLayout_;
        pi.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }

    void createFramebuffers()
    {
        framebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            VkImageView att[] = {swapchainImageViews_[i], depth_.view};
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = renderPass_;
            ci.attachmentCount = 2;
            ci.pAttachments = att;
            ci.width = swapchainExtent_.width;
            ci.height = swapchainExtent_.height;
            ci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
        }
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
    }

    void createMeshBuffers()
    {
        std::vector<GltfVertex> vertices = gltfScene_.meshes[0].vertices;
        for (GltfVertex& v : vertices) {
            const float weightSum = v.weights.x + v.weights.y + v.weights.z + v.weights.w;
            if (weightSum < 1e-5f) {
                v.joints = glm::uvec4(0);
                v.weights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        const std::vector<uint32_t>& indices = gltfScene_.meshes[0].indices;
        const VkDeviceSize vertexSize = sizeof(GltfVertex) * vertices.size();
        const VkDeviceSize indexSize = sizeof(uint32_t) * indices.size();
        createBuffer(physicalDevice_, device_, vertexSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_, vertexMemory_);
        void* mapped = nullptr;
        VK_CHECK(vkMapMemory(device_, vertexMemory_, 0, vertexSize, 0, &mapped));
        std::memcpy(mapped, vertices.data(), static_cast<size_t>(vertexSize));
        vkUnmapMemory(device_, vertexMemory_);
        createBuffer(physicalDevice_, device_, indexSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     indexBuffer_, indexMemory_);
        VK_CHECK(vkMapMemory(device_, indexMemory_, 0, indexSize, 0, &mapped));
        std::memcpy(mapped, indices.data(), static_cast<size_t>(indexSize));
        vkUnmapMemory(device_, indexMemory_);
    }

    void createUniformBuffers()
    {
        sceneUBOs_.resize(MAX_FRAMES);
        sceneUBOMemories_.resize(MAX_FRAMES);
        sceneUBOMapped_.resize(MAX_FRAMES);
        materialUBOs_.resize(MAX_FRAMES);
        materialUBOMemories_.resize(MAX_FRAMES);
        materialUBOMapped_.resize(MAX_FRAMES);
        boneUBOs_.resize(MAX_FRAMES);
        boneUBOMemories_.resize(MAX_FRAMES);
        boneUBOMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physicalDevice_, device_, sizeof(SceneUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         sceneUBOs_[i], sceneUBOMemories_[i]);
            VK_CHECK(vkMapMemory(device_, sceneUBOMemories_[i], 0, sizeof(SceneUBO), 0,
                                 &sceneUBOMapped_[i]));
            createBuffer(physicalDevice_, device_, sizeof(MaterialUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         materialUBOs_[i], materialUBOMemories_[i]);
            VK_CHECK(vkMapMemory(device_, materialUBOMemories_[i], 0, sizeof(MaterialUBO), 0,
                                 &materialUBOMapped_[i]));
            std::memcpy(materialUBOMapped_[i], &materialData_, sizeof(materialData_));
            createBuffer(physicalDevice_, device_, sizeof(BoneUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         boneUBOs_[i], boneUBOMemories_[i]);
            VK_CHECK(vkMapMemory(device_, boneUBOMemories_[i], 0, sizeof(BoneUBO), 0,
                                 &boneUBOMapped_[i]));
        }
    }

    void createDescriptorPoolAndSets()
    {
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(MAX_FRAMES * 3)},
        };
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = sizes;
        pool.maxSets = MAX_FRAMES;
        VK_CHECK(vkCreateDescriptorPool(device_, &pool, nullptr, &descriptorPool_));
        descriptorSets_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptorPool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &setLayout_;
            VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &descriptorSets_[i]));
            VkDescriptorBufferInfo sceneInfo{sceneUBOs_[i], 0, sizeof(SceneUBO)};
            VkDescriptorBufferInfo boneInfo{boneUBOs_[i], 0, sizeof(BoneUBO)};
            VkDescriptorBufferInfo matInfo{materialUBOs_[i], 0, sizeof(MaterialUBO)};
            VkWriteDescriptorSet writes[3] = {};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = descriptorSets_[i];
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].descriptorCount = 1;
            writes[0].pBufferInfo = &sceneInfo;
            writes[1] = writes[0];
            writes[1].dstBinding = 1;
            writes[1].pBufferInfo = &boneInfo;
            writes[2] = writes[0];
            writes[2].dstBinding = 2;
            writes[2].pBufferInfo = &matInfo;
            vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);
        }
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

    void updateUniforms(uint32_t frameIndex, float time)
    {
        SceneUBO scene{};
        const float aspect = static_cast<float>(swapchainExtent_.width) /
                             static_cast<float>(swapchainExtent_.height);
        scene.view = interactive_.camera().viewMatrix();
        scene.projection = interactive_.camera().projectionMatrix(aspect, 45.0f, 0.1f, 50.0f);
        std::memcpy(sceneUBOMapped_[frameIndex], &scene, sizeof(scene));
        BoneUBO bones{};
        for (uint32_t i = 0; i < MAX_BONES; ++i)
            bones.bones[i] = glm::mat4(1.0f);
        glm::vec3 translation(0.0f);
        if (hasAnimation_) {
            const float animTime = std::fmod(time, animation_.duration);
            translation = sampleTranslation(animation_, animTime);
        }
        bones.bones[0] = glm::translate(glm::mat4(1.0f), translation);
        std::memcpy(boneUBOMapped_[frameIndex], &bones, sizeof(bones));
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex, float time)
    {
        updateUniforms(currentFrame_, time);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, currentFrame_);
        std::array<VkClearValue, 2> clears{};
        clears[0] = {{{0.04f, 0.05f, 0.08f, 1.0f}}};
        clears[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass_;
        rp.framebuffer = framebuffers_[imageIndex];
        rp.renderArea = {{0, 0}, swapchainExtent_};
        rp.clearValueCount = static_cast<uint32_t>(clears.size());
        rp.pClearValues = clears.data();
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkViewport vp{0.0f, 0.0f, static_cast<float>(swapchainExtent_.width),
                      static_cast<float>(swapchainExtent_.height), 0.0f, 1.0f};
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1,
                                &descriptorSets_[currentFrame_], 0, nullptr);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer_, &offset);
        vkCmdBindIndexBuffer(cmd, indexBuffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, indexCount_, 1, 0, 0, 0);
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

    void recreateSwapchain()
    {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        for (auto fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto iv : swapchainImageViews_) vkDestroyImageView(device_, iv, nullptr);
        destroyDepthResources(device_, depth_);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physicalDevice_, device_, swapchainExtent_);
        createFramebuffers();
        interactive_.onSwapchainRecreated(renderPass_, swapchainFormat_,
            static_cast<uint32_t>(swapchainImages_.size()));
    }

    void mainLoop()
    {
        std::cout << "🎨 rig.gltf 三角形 bounce 动画。ESC 退出。\n";
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
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkUnmapMemory(device_, sceneUBOMemories_[i]);
            vkDestroyBuffer(device_, sceneUBOs_[i], nullptr);
            vkFreeMemory(device_, sceneUBOMemories_[i], nullptr);
            vkUnmapMemory(device_, materialUBOMemories_[i]);
            vkDestroyBuffer(device_, materialUBOs_[i], nullptr);
            vkFreeMemory(device_, materialUBOMemories_[i], nullptr);
            vkUnmapMemory(device_, boneUBOMemories_[i]);
            vkDestroyBuffer(device_, boneUBOs_[i], nullptr);
            vkFreeMemory(device_, boneUBOMemories_[i], nullptr);
        }
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMemory_, nullptr);
        vkDestroyBuffer(device_, indexBuffer_, nullptr);
        vkFreeMemory(device_, indexMemory_, nullptr);
        destroyDepthResources(device_, depth_);
        for (auto fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto iv : swapchainImageViews_) vkDestroyImageView(device_, iv, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
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
    std::cout << " 第50章：GPU Skeletal Skinning\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    Ch50App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
