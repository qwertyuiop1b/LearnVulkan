/**
 * @file ch41_skybox.cpp
 * @brief 第41章：立方体贴图天空盒（Cubemap Skybox）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【天空盒原理】
 *
 *  将 6 张环境贴图（+X/-X/+Y/-Y/+Z/-Z）组合为立方体贴图，
 *  在顶点着色器中移除 view 矩阵的平移分量，并将深度固定为 1.0，
 *  使天空盒始终渲染在场景最远处。
 *
 * 【深度测试配置】
 *
 *  depthCompareOp = LESS_OR_EQUAL，depthWriteEnable = FALSE
 *  gl_Position.xyww 使片段深度恒为 1.0（最远平面）
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
#include <set>
#include <stdexcept>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;
constexpr uint32_t CUBEMAP_SIZE = 256;

struct SkyboxUBO {
    alignas(16) glm::mat4 projection;
    alignas(16) glm::mat4 view;
};

struct CubemapTexture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
};

struct SkyboxVertex {
    glm::vec3 pos;
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription d{};
        d.binding = 0;
        d.stride = sizeof(SkyboxVertex);
        d.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return d;
    }
    static std::array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 1> a{};
        a[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SkyboxVertex, pos)};
        return a;
    }
};

static const std::vector<SkyboxVertex> SKYBOX_VERTICES = {
    {{-1.0f, -1.0f, -1.0f}}, {{1.0f, -1.0f, -1.0f}},  {{1.0f, 1.0f, -1.0f}},   {{1.0f, 1.0f, -1.0f}},
    {{-1.0f, 1.0f, -1.0f}},  {{-1.0f, -1.0f, -1.0f}}, {{-1.0f, -1.0f, 1.0f}},  {{1.0f, -1.0f, 1.0f}},
    {{1.0f, 1.0f, 1.0f}},    {{1.0f, 1.0f, 1.0f}},    {{-1.0f, 1.0f, 1.0f}},   {{-1.0f, -1.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}},   {{-1.0f, 1.0f, -1.0f}},  {{-1.0f, -1.0f, -1.0f}}, {{-1.0f, -1.0f, -1.0f}},
    {{-1.0f, -1.0f, 1.0f}},  {{-1.0f, 1.0f, 1.0f}},   {{1.0f, 1.0f, 1.0f}},    {{1.0f, 1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f}},  {{1.0f, -1.0f, -1.0f}},  {{1.0f, -1.0f, 1.0f}},   {{1.0f, 1.0f, 1.0f}},
    {{-1.0f, -1.0f, -1.0f}}, {{1.0f, -1.0f, -1.0f}},  {{1.0f, -1.0f, 1.0f}},   {{1.0f, -1.0f, 1.0f}},
    {{-1.0f, -1.0f, 1.0f}},  {{-1.0f, -1.0f, -1.0f}}, {{-1.0f, 1.0f, -1.0f}},  {{1.0f, 1.0f, -1.0f}},
    {{1.0f, 1.0f, 1.0f}},    {{1.0f, 1.0f, 1.0f}},    {{-1.0f, 1.0f, 1.0f}},   {{-1.0f, 1.0f, -1.0f}},
};

static ImageData generateCubemapFace(uint32_t faceIndex, uint32_t size) {
    ImageData img;
    img.width = size;
    img.height = size;
    img.channels = 4;
    img.pixels.resize(static_cast<size_t>(size * size * 4));
    const glm::vec3 faceColors[6] = {{0.9f, 0.3f, 0.3f},
                                     {0.3f, 0.9f, 0.3f},
                                     {0.3f, 0.3f, 0.9f},
                                     {0.9f, 0.9f, 0.3f},
                                     {0.3f, 0.9f, 0.9f},
                                     {0.9f, 0.3f, 0.9f}};
    const glm::vec3 base = faceColors[faceIndex % 6];
    for (uint32_t y = 0; y < size; ++y) {
        const float v = static_cast<float>(y) / static_cast<float>(size);
        for (uint32_t x = 0; x < size; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(size);
            const size_t i = static_cast<size_t>((y * size + x) * 4);
            img.pixels[i + 0] = static_cast<uint8_t>((base.r * (0.5f + 0.5f * u)) * 255.0f);
            img.pixels[i + 1] = static_cast<uint8_t>((base.g * (0.5f + 0.5f * v)) * 255.0f);
            img.pixels[i + 2] = static_cast<uint8_t>((base.b * (0.5f + 0.5f * (u + v) * 0.5f)) * 255.0f);
            img.pixels[i + 3] = 255;
        }
    }
    return img;
}

static void copyBufferToImageLayer(VkDevice device,
                                   VkCommandPool pool,
                                   VkQueue queue,
                                   VkBuffer buffer,
                                   VkImage image,
                                   uint32_t width,
                                   uint32_t height,
                                   uint32_t layer) {
    VkCommandBuffer cmd = beginSingleTimeCommands(device, pool);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = layer;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    endSingleTimeCommands(device, pool, queue, cmd);
}

static CubemapTexture createCubemapFromFaces(VkPhysicalDevice physicalDevice,
                                             VkDevice device,
                                             VkCommandPool pool,
                                             VkQueue queue,
                                             const std::array<ImageData, 6>& faces) {
    CubemapTexture cubemap{};
    const uint32_t size = faces[0].width;
    cubemap.format = VK_FORMAT_R8G8B8A8_SRGB;
    createImage(physicalDevice,
                device,
                size,
                size,
                cubemap.format,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                cubemap.image,
                cubemap.memory,
                1,
                6,
                VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
    transitionImageLayout(device,
                          pool,
                          queue,
                          cubemap.image,
                          cubemap.format,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    for (uint32_t face = 0; face < 6; ++face) {
        const VkDeviceSize imageSize =
            static_cast<VkDeviceSize>(faces[face].width * faces[face].height * faces[face].channels);
        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        createBuffer(physicalDevice,
                     device,
                     imageSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer,
                     stagingMemory);
        void* mapped = nullptr;
        vkMapMemory(device, stagingMemory, 0, imageSize, 0, &mapped);
        std::memcpy(mapped, faces[face].pixels.data(), static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingMemory);
        copyBufferToImageLayer(
            device, pool, queue, stagingBuffer, cubemap.image, faces[face].width, faces[face].height, face);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }
    transitionImageLayout(device,
                          pool,
                          queue,
                          cubemap.image,
                          cubemap.format,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = cubemap.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format = cubemap.format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 6;
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &cubemap.view));
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &cubemap.sampler));
    return cubemap;
}

static void destroyCubemap(VkDevice device, CubemapTexture& cubemap) {
    if (cubemap.sampler != VK_NULL_HANDLE)
        vkDestroySampler(device, cubemap.sampler, nullptr);
    if (cubemap.view != VK_NULL_HANDLE)
        vkDestroyImageView(device, cubemap.view, nullptr);
    if (cubemap.image != VK_NULL_HANDLE)
        vkDestroyImage(device, cubemap.image, nullptr);
    if (cubemap.memory != VK_NULL_HANDLE)
        vkFreeMemory(device, cubemap.memory, nullptr);
    cubemap = {};
}

class Ch41App {
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
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    DepthResources depth_;
    CubemapTexture skyboxCubemap_;
    std::vector<VkDescriptorSet> descriptorSets_;
    std::vector<VkBuffer> uniformBuffers_;
    std::vector<VkDeviceMemory> uniformBuffersMemory_;
    std::vector<void*> uniformBuffersMapped_;
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    QueueFamilyIndices queueIndices_;
    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory_ = VK_NULL_HANDLE;
    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;
    InteractiveChapterTools interactive_;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch41 - 立方体贴图天空盒", nullptr, nullptr);
        interactive_.attachInput(window_);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch41App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan() {
        vulkan_tutorial::createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physicalDevice_);
        createLogicalDevice(physicalDevice_, surface_, device_, graphicsQueue_, presentQueue_, queueIndices_);
        createSwapchain();
        createImageViews();
        depth_.format = findDepthFormat(physicalDevice_);
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        depth_ = createDepthResources(physicalDevice_, device_, swapchainExtent_);
        createFramebuffers();
        createCommandPool();
        createSkyboxMesh();
        createSkyboxCubemap();
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
        std::cout << "\n✅ 天空盒初始化完成！\n";
    }

    void createSkyboxCubemap() {
        std::array<ImageData, 6> faces{};
        for (uint32_t i = 0; i < 6; ++i)
            faces[i] = generateCubemapFace(i, CUBEMAP_SIZE);
        skyboxCubemap_ = createCubemapFromFaces(physicalDevice_, device_, commandPool_, graphicsQueue_, faces);
        std::cout << "✅ 程序化立方体贴图已创建（6 面 × " << CUBEMAP_SIZE << "²）\n";
    }

    void createSkyboxMesh() {
        const VkDeviceSize size = sizeof(SKYBOX_VERTICES[0]) * SKYBOX_VERTICES.size();
        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        createBuffer(physicalDevice_,
                     device_,
                     size,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer,
                     stagingMemory);
        void* data = nullptr;
        vkMapMemory(device_, stagingMemory, 0, size, 0, &data);
        std::memcpy(data, SKYBOX_VERTICES.data(), static_cast<size_t>(size));
        vkUnmapMemory(device_, stagingMemory);
        createBuffer(physicalDevice_,
                     device_,
                     size,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     vertexBuffer_,
                     vertexBufferMemory_);
        copyBuffer(device_, commandPool_, graphicsQueue_, stagingBuffer, vertexBuffer_, size);
        vkDestroyBuffer(device_, stagingBuffer, nullptr);
        vkFreeMemory(device_, stagingMemory, nullptr);
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapchainImageFormat_;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = depth_.format;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = static_cast<uint32_t>(attachments.size());
        rpi.pAttachments = attachments.data();
        rpi.subpassCount = 1;
        rpi.pSubpasses = &subpass;
        rpi.dependencyCount = 1;
        rpi.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_));
    }

    void createDescriptorSetLayout() {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = static_cast<uint32_t>(bindings.size());
        ci.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descriptorSetLayout_));
    }

    void createGraphicsPipeline() {
        VkShaderModule vert = createShaderModuleFromFile(device_, "skybox.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "skybox.frag.spv");
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
        auto bd = SkyboxVertex::getBindingDescription();
        auto ad = SkyboxVertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bd;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(ad.size());
        vi.pVertexAttributeDescriptions = ad.data();
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
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        ds.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        ds.pDynamicStates = dyn.data();
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &descriptorSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &pipelineLayout_));
        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount = 2;
        pi.pStages = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vs;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &depthStencil;
        pi.pColorBlendState = &cb;
        pi.pDynamicState = &ds;
        pi.layout = pipelineLayout_;
        pi.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }

    void createFramebuffers() {
        framebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            std::array<VkImageView, 2> attachments = {swapchainImageViews_[i], depth_.view};
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = renderPass_;
            ci.attachmentCount = static_cast<uint32_t>(attachments.size());
            ci.pAttachments = attachments.data();
            ci.width = swapchainExtent_.width;
            ci.height = swapchainExtent_.height;
            ci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
        }
    }

    void createUniformBuffers() {
        const VkDeviceSize size = sizeof(SkyboxUBO);
        uniformBuffers_.resize(MAX_FRAMES);
        uniformBuffersMemory_.resize(MAX_FRAMES);
        uniformBuffersMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physicalDevice_,
                         device_,
                         size,
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers_[i],
                         uniformBuffersMemory_[i]);
            vkMapMemory(device_, uniformBuffersMemory_[i], 0, size, 0, &uniformBuffersMapped_[i]);
        }
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> sizes{};
        sizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(MAX_FRAMES)};
        sizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(MAX_FRAMES)};
        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        ci.pPoolSizes = sizes.data();
        ci.maxSets = static_cast<uint32_t>(MAX_FRAMES);
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &descriptorPool_));
    }

    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES, descriptorSetLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES);
        ai.pSetLayouts = layouts.data();
        descriptorSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, descriptorSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bufInfo{};
            bufInfo.buffer = uniformBuffers_[i];
            bufInfo.offset = 0;
            bufInfo.range = sizeof(SkyboxUBO);
            VkDescriptorImageInfo imgInfo{};
            imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imgInfo.imageView = skyboxCubemap_.view;
            imgInfo.sampler = skyboxCubemap_.sampler;
            std::array<VkWriteDescriptorSet, 2> writes{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = descriptorSets_[i];
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].descriptorCount = 1;
            writes[0].pBufferInfo = &bufInfo;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = descriptorSets_[i];
            writes[1].dstBinding = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].descriptorCount = 1;
            writes[1].pImageInfo = &imgInfo;
            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t idx) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, currentFrame_);
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.02f, 0.02f, 0.05f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo rpBI{};
        rpBI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBI.renderPass = renderPass_;
        rpBI.framebuffer = framebuffers_[idx];
        rpBI.renderArea = {{0, 0}, swapchainExtent_};
        rpBI.clearValueCount = static_cast<uint32_t>(clearValues.size());
        rpBI.pClearValues = clearValues.data();
        vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkViewport vp{0, 0, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &sc);
        VkBuffer vb[] = {vertexBuffer_};
        VkDeviceSize off[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &descriptorSets_[currentFrame_], 0, nullptr);
        vkCmdDraw(cmd, static_cast<uint32_t>(SKYBOX_VERTICES.size()), 1, 0, 0);
        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);
        interactive_.endGpuSection(cmd, currentFrame_);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void updateUniformBuffer(uint32_t frame) {
        SkyboxUBO ubo{};
        const float aspect = static_cast<float>(swapchainExtent_.width) / static_cast<float>(swapchainExtent_.height);
        ubo.projection = interactive_.camera().projectionMatrix(aspect, 45.0f, 0.1f, 10.0f);
        ubo.view = interactive_.camera().viewMatrixWithoutTranslation();
        std::memcpy(uniformBuffersMapped_[frame], &ubo, sizeof(ubo));
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
        updateUniformBuffer(currentFrame_);
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
        VkSwapchainKHR sc2[] = {swapchain_};
        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = ss;
        pi.swapchainCount = 1;
        pi.pSwapchains = sc2;
        pi.pImageIndices = &imgIdx;
        r = vkQueuePresentKHR(presentQueue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        }
        interactive_.endFrame(currentFrame_);
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void mainLoop() {
        std::cout << "🎨 旋转相机观察程序化天空盒（深度=1.0）...\n";
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

    void recreateSwapchain() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (!w || !h) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        for (auto& fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        destroyDepthResources(device_, depth_);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physicalDevice_, device_, swapchainExtent_);
        createFramebuffers();
        interactive_.onSwapchainRecreated(
            renderPass_, swapchainImageFormat_, static_cast<uint32_t>(swapchainImages_.size()));
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

    void cleanup() {
        destroyCubemap(device_, skyboxCubemap_);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, uniformBuffers_[i], nullptr);
            vkFreeMemory(device_, uniformBuffersMemory_[i], nullptr);
        }
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexBufferMemory_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (auto& fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        destroyDepthResources(device_, depth_);
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        for (auto& iv : swapchainImageViews_)
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

int main() {
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << " 第41章：立方体贴图天空盒\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    Ch41App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
