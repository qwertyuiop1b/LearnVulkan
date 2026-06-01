/**
 * @file ch47_ssr.cpp
 * @brief 第47章：屏幕空间反射（SSR）+ Cubemap 降级
 *
 * 管线：场景（颜色+法线+深度）→ SSR 追踪 → 与 Cubemap 降级合成 → 呈现。
 * 屏幕空间步进失败时采样程序生成的 Cubemap 作为反射降级。
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
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;

struct GVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec2 uv;
};

static const std::vector<GVertex> SCENE = {
    {{-3, -0.5f, -3}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}, {0, 0}},
    {{3, -0.5f, -3}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}, {1, 0}},
    {{3, -0.5f, 3}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}, {1, 1}},
    {{-3, -0.5f, -3}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}, {0, 0}},
    {{3, -0.5f, 3}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}, {1, 1}},
    {{-3, -0.5f, 3}, {0, 1, 0}, {0.45f, 0.45f, 0.5f}, {0, 1}},
    {{-0.5f, -0.48f, -0.5f}, {0, 0, -1}, {0.75f, 0.35f, 0.35f}, {0, 0}},
    {{0.5f, -0.48f, -0.5f}, {0, 0, -1}, {0.75f, 0.35f, 0.35f}, {1, 0}},
    {{0.5f, 1.2f, -0.5f}, {0, 0, -1}, {0.75f, 0.35f, 0.35f}, {1, 1}},
    {{-0.5f, -0.48f, -0.5f}, {0, 0, -1}, {0.75f, 0.35f, 0.35f}, {0, 0}},
    {{0.5f, 1.2f, -0.5f}, {0, 0, -1}, {0.75f, 0.35f, 0.35f}, {1, 1}},
    {{-0.5f, 1.2f, -0.5f}, {0, 0, -1}, {0.75f, 0.35f, 0.35f}, {0, 1}},
    {{1.0f, -0.5f, 0.2f}, {1, 0, 0}, {0.35f, 0.45f, 0.9f}, {0, 0}},
    {{1.4f, -0.5f, 0.2f}, {1, 0, 0}, {0.35f, 0.45f, 0.9f}, {1, 0}},
    {{1.4f, 1.5f, 0.2f}, {1, 0, 0}, {0.35f, 0.45f, 0.9f}, {1, 1}},
    {{1.0f, -0.5f, 0.2f}, {1, 0, 0}, {0.35f, 0.45f, 0.9f}, {0, 0}},
    {{1.4f, 1.5f, 0.2f}, {1, 0, 0}, {0.35f, 0.45f, 0.9f}, {1, 1}},
    {{1.0f, 1.5f, 0.2f}, {1, 0, 0}, {0.35f, 0.45f, 0.9f}, {0, 1}},
};

struct GeomUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};
struct SSRUBO {
    alignas(16) glm::mat4 invView;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::vec4 cameraPos;
    float maxDistance;
    float stepSize;
    float thickness;
    float intensity;
};

class Ch47App {
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

    VkImage posImage_ = VK_NULL_HANDLE;
    VkDeviceMemory posMem_ = VK_NULL_HANDLE;
    VkImageView posView_ = VK_NULL_HANDLE;
    VkImage normalImage_ = VK_NULL_HANDLE;
    VkDeviceMemory normalMem_ = VK_NULL_HANDLE;
    VkImageView normalView_ = VK_NULL_HANDLE;
    VkImage colorImage_ = VK_NULL_HANDLE;
    VkDeviceMemory colorMem_ = VK_NULL_HANDLE;
    VkImageView colorView_ = VK_NULL_HANDLE;
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthMem_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;
    VkImage reflImage_ = VK_NULL_HANDLE;
    VkDeviceMemory reflMem_ = VK_NULL_HANDLE;
    VkImageView reflView_ = VK_NULL_HANDLE;
    VkImage cubeImage_ = VK_NULL_HANDLE;
    VkDeviceMemory cubeMem_ = VK_NULL_HANDLE;
    VkImageView cubeView_ = VK_NULL_HANDLE;
    VkSampler linearSampler_ = VK_NULL_HANDLE;
    VkSampler cubeSampler_ = VK_NULL_HANDLE;

    VkRenderPass sceneRP_ = VK_NULL_HANDLE;
    VkRenderPass ssrRP_ = VK_NULL_HANDLE;
    VkRenderPass compositeRP_ = VK_NULL_HANDLE;
    VkFramebuffer sceneFB_ = VK_NULL_HANDLE;
    VkFramebuffer ssrFB_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> compositeFBs_;

    VkPipeline geomPipeline_ = VK_NULL_HANDLE;
    VkPipeline ssrPipeline_ = VK_NULL_HANDLE;
    VkPipeline compositePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout geomLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout ssrLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout compositeLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout geomDSL_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssrDSL_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout compositeDSL_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> geomSets_;
    std::vector<VkDescriptorSet> ssrSets_;
    std::vector<VkDescriptorSet> compositeSets_;

    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMem_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> geomUBOs_;
    std::vector<VkDeviceMemory> geomUBOMem_;
    std::vector<void*> geomMapped_;
    std::vector<VkBuffer> ssrUBOs_;
    std::vector<VkDeviceMemory> ssrUBOMem_;
    std::vector<void*> ssrMapped_;

    std::vector<VkImage> swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<VkCommandBuffer> cmdBuffers_;
    std::vector<VkSemaphore> imageAvailable_;
    std::vector<VkSemaphore> renderFinished_;
    std::vector<VkFence> inFlight_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;
    InteractiveChapterTools interactive_;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch47 - SSR + Cubemap 降级", nullptr, nullptr);
        interactive_.attachInput(window_);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch47App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan() {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physicalDevice_);
        createLogicalDevice(physicalDevice_, surface_, device_, graphicsQueue_, presentQueue_, queueIndices_);
        createSwapchain();
        createImageViews();
        depthFormat_ = findDepthStencilFormat(physicalDevice_);
        if (depthFormat_ == VK_FORMAT_UNDEFINED)
            depthFormat_ = findDepthFormat(physicalDevice_);
        createCommandPool();
        createSceneTargets();
        createReflectionTarget();
        createFallbackCubemap();
        createSamplers();
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
        interactive_.camera().setTarget(glm::vec3(0.0f, 0.3f, 0.0f));
        interactive_.camera().setDistance(6.0f);
        std::cout << "✅ SSR 初始化完成（Cubemap 降级已就绪）\n";
    }

    void createSceneTargets() {
        const VkImageUsageFlags colorUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        createImage(physicalDevice_,
                    device_,
                    extent_.width,
                    extent_.height,
                    VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_TILING_OPTIMAL,
                    colorUsage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    posImage_,
                    posMem_);
        createImage(physicalDevice_,
                    device_,
                    extent_.width,
                    extent_.height,
                    VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_TILING_OPTIMAL,
                    colorUsage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    normalImage_,
                    normalMem_);
        createImage(physicalDevice_,
                    device_,
                    extent_.width,
                    extent_.height,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_TILING_OPTIMAL,
                    colorUsage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    colorImage_,
                    colorMem_);
        createImage(physicalDevice_,
                    device_,
                    extent_.width,
                    extent_.height,
                    depthFormat_,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    depthImage_,
                    depthMem_);
        auto makeView = [&](VkImage img, VkFormat fmt, VkImageAspectFlags aspect, VkImageView& view) {
            VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            ci.image = img;
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = fmt;
            ci.subresourceRange = {aspect, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &view));
        };
        makeView(posImage_, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, posView_);
        makeView(normalImage_, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, normalView_);
        makeView(colorImage_, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, colorView_);
        makeView(depthImage_, depthFormat_, VK_IMAGE_ASPECT_DEPTH_BIT, depthView_);
    }

    void createReflectionTarget() {
        createImage(physicalDevice_,
                    device_,
                    extent_.width,
                    extent_.height,
                    VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    reflImage_,
                    reflMem_);
        VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ci.image = reflImage_;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &reflView_));
    }

    void createFallbackCubemap() {
        const uint32_t faceSize = 64;
        ImageData face = generateCheckerboard(faceSize, 8);
        const VkDeviceSize faceBytes = faceSize * faceSize * 4;
        const VkDeviceSize totalBytes = faceBytes * 6;
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory stagingMem = VK_NULL_HANDLE;
        createBuffer(physicalDevice_,
                     device_,
                     totalBytes,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging,
                     stagingMem);
        void* mapped = nullptr;
        vkMapMemory(device_, stagingMem, 0, totalBytes, 0, &mapped);
        for (uint32_t f = 0; f < 6; ++f) {
            uint8_t* dst = static_cast<uint8_t*>(mapped) + f * faceBytes;
            for (uint32_t i = 0; i < faceSize * faceSize; ++i) {
                dst[i * 4 + 0] = static_cast<uint8_t>(
                    std::min(255, static_cast<int>(face.pixels[i * 4 + 0]) + static_cast<int>(f) * 12));
                dst[i * 4 + 1] = static_cast<uint8_t>(std::min(255, static_cast<int>(face.pixels[i * 4 + 1]) + 20));
                dst[i * 4 + 2] = static_cast<uint8_t>(std::min(255, static_cast<int>(face.pixels[i * 4 + 2]) + 40));
                dst[i * 4 + 3] = 255;
            }
        }
        vkUnmapMemory(device_, stagingMem);
        createImage(physicalDevice_,
                    device_,
                    faceSize,
                    faceSize,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    cubeImage_,
                    cubeMem_,
                    1,
                    6,
                    VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_, commandPool_);
        VkImageMemoryBarrier toDst{};
        toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcQueueFamilyIndex = toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image = cubeImage_;
        toDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
        toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &toDst);
        for (uint32_t face = 0; face < 6; ++face) {
            VkBufferImageCopy region{};
            region.bufferOffset = face * faceBytes;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = face;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {faceSize, faceSize, 1};
            vkCmdCopyBufferToImage(cmd, staging, cubeImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        VkImageMemoryBarrier toRead = toDst;
        toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &toRead);
        endSingleTimeCommands(device_, commandPool_, graphicsQueue_, cmd);
        vkDestroyBuffer(device_, staging, nullptr);
        vkFreeMemory(device_, stagingMem, nullptr);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = cubeImage_;
        vi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        vi.format = VK_FORMAT_R8G8B8A8_UNORM;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
        VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &cubeView_));
    }

    void createSamplers() {
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter = ci.minFilter = VK_FILTER_LINEAR;
        ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &linearSampler_));
        ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &cubeSampler_));
    }

    void createRenderPasses() {
        VkAttachmentDescription posAtt{};
        posAtt.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        posAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        posAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        posAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        posAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        posAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        posAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        posAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentDescription normalAtt = posAtt;
        VkAttachmentDescription colorAtt = posAtt;
        colorAtt.format = VK_FORMAT_R8G8B8A8_UNORM;
        VkAttachmentDescription depth{};
        depth.format = depthFormat_;
        depth.samples = VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        std::array<VkAttachmentReference, 3> colorRefs = {{{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
                                                           {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
                                                           {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}}};
        VkAttachmentReference depthRef{3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 3;
        sub.pColorAttachments = colorRefs.data();
        sub.pDepthStencilAttachment = &depthRef;
        std::array<VkAttachmentDescription, 4> atts = {posAtt, normalAtt, colorAtt, depth};
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.attachmentCount = static_cast<uint32_t>(atts.size());
        rp.pAttachments = atts.data();
        rp.subpassCount = 1;
        rp.pSubpasses = &sub;
        VK_CHECK(vkCreateRenderPass(device_, &rp, nullptr, &sceneRP_));

        VkAttachmentDescription reflAtt = posAtt;
        reflAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentReference reflRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription ssrSub{};
        ssrSub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        ssrSub.colorAttachmentCount = 1;
        ssrSub.pColorAttachments = &reflRef;
        rp.attachmentCount = 1;
        rp.pAttachments = &reflAtt;
        rp.pSubpasses = &ssrSub;
        VK_CHECK(vkCreateRenderPass(device_, &rp, nullptr, &ssrRP_));

        VkAttachmentDescription outAtt = colorAtt;
        outAtt.format = swapFormat_;
        outAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference outRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription compSub{};
        compSub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        compSub.colorAttachmentCount = 1;
        compSub.pColorAttachments = &outRef;
        rp.pAttachments = &outAtt;
        rp.pSubpasses = &compSub;
        VK_CHECK(vkCreateRenderPass(device_, &rp, nullptr, &compositeRP_));
    }

    VkShaderModule loadShader(const char* name) {
        return createShaderModuleFromFile(device_, name);
    }

    VkPipeline buildFullscreenPipeline(VkRenderPass rp, VkPipelineLayout layout, const char* fragName) {
        VkShaderModule vert = loadShader("deferred_lighting.vert.spv");
        VkShaderModule fragMod = loadShader(fragName);
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
                                                      fragMod,
                                                      "main",
                                                      nullptr}};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
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
        vkDestroyShaderModule(device_, fragMod, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        return pipe;
    }

    void createDescriptorLayouts() {
        VkDescriptorSetLayoutBinding gBind{
            0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 1;
        ci.pBindings = &gBind;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &geomDSL_));
        std::array<VkDescriptorSetLayoutBinding, 5> sBinds = {
            {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
             {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
             {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
             {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
             {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
        ci.bindingCount = static_cast<uint32_t>(sBinds.size());
        ci.pBindings = sBinds.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &ssrDSL_));
        std::array<VkDescriptorSetLayoutBinding, 2> cBinds = {
            {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
             {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
        ci.bindingCount = static_cast<uint32_t>(cBinds.size());
        ci.pBindings = cBinds.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &compositeDSL_));
    }

    void createPipelines() {
        VkShaderModule gVert = loadShader("gbuffer.vert.spv");
        VkShaderModule gFrag = loadShader("gbuffer.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                      nullptr,
                                                      0,
                                                      VK_SHADER_STAGE_VERTEX_BIT,
                                                      gVert,
                                                      "main",
                                                      nullptr},
                                                     {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                      nullptr,
                                                      0,
                                                      VK_SHADER_STAGE_FRAGMENT_BIT,
                                                      gFrag,
                                                      "main",
                                                      nullptr}};
        VkVertexInputBindingDescription bind{0, sizeof(GVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 4> attrs{};
        attrs[0].location = 0;
        attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(GVertex, pos);
        attrs[1].location = 1;
        attrs[1].binding = 0;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = offsetof(GVertex, normal);
        attrs[2].location = 2;
        attrs[2].binding = 0;
        attrs[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[2].offset = offsetof(GVertex, color);
        attrs[3].location = 3;
        attrs[3].binding = 0;
        attrs[3].format = VK_FORMAT_R32G32_SFLOAT;
        attrs[3].offset = offsetof(GVertex, uv);
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
        std::array<VkPipelineColorBlendAttachmentState, 3> cbas{};
        cbas[0].colorWriteMask = cbas[1].colorWriteMask = cbas[2].colorWriteMask = 0xF;
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
        gp.pStages = stages;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vps;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cb;
        gp.pDynamicState = &dynS;
        gp.layout = geomLayout_;
        gp.renderPass = sceneRP_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &geomPipeline_));
        vkDestroyShaderModule(device_, gFrag, nullptr);
        vkDestroyShaderModule(device_, gVert, nullptr);

        pl.pSetLayouts = &ssrDSL_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pl, nullptr, &ssrLayout_));
        ssrPipeline_ = buildFullscreenPipeline(ssrRP_, ssrLayout_, "ssr.frag.spv");

        pl.pSetLayouts = &compositeDSL_;
        VkPushConstantRange pc{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device_, &pl, nullptr, &compositeLayout_));
        compositePipeline_ = buildFullscreenPipeline(compositeRP_, compositeLayout_, "ssr_composite.frag.spv");
    }

    void createFramebuffers() {
        std::array<VkImageView, 4> sceneAtt = {posView_, normalView_, colorView_, depthView_};
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = sceneRP_;
        fb.attachmentCount = static_cast<uint32_t>(sceneAtt.size());
        fb.pAttachments = sceneAtt.data();
        fb.width = extent_.width;
        fb.height = extent_.height;
        fb.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &fb, nullptr, &sceneFB_));
        fb.renderPass = ssrRP_;
        fb.attachmentCount = 1;
        fb.pAttachments = &reflView_;
        VK_CHECK(vkCreateFramebuffer(device_, &fb, nullptr, &ssrFB_));
        compositeFBs_.resize(swapViews_.size());
        for (size_t i = 0; i < swapViews_.size(); ++i) {
            fb.renderPass = compositeRP_;
            fb.pAttachments = &swapViews_[i];
            VK_CHECK(vkCreateFramebuffer(device_, &fb, nullptr, &compositeFBs_[i]));
        }
    }

    void updateUniforms(uint32_t frame) {
        GeomUBO g{};
        g.model = glm::mat4(1.0f);
        const float aspect = extent_.width / static_cast<float>(extent_.height);
        g.view = interactive_.camera().viewMatrix();
        g.projection = interactive_.camera().projectionMatrix(aspect, 55.0f, 0.1f, 40.0f);
        std::memcpy(geomMapped_[frame], &g, sizeof(g));
        SSRUBO s{};
        s.view = g.view;
        s.invView = glm::inverse(g.view);
        s.projection = g.projection;
        s.cameraPos = glm::vec4(interactive_.camera().eyePosition(), 1.0f);
        s.maxDistance = 12.0f;
        s.stepSize = 0.15f;
        s.thickness = 0.25f;
        s.intensity = 0.85f;
        std::memcpy(ssrMapped_[frame], &s, sizeof(s));
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginInfo);
        interactive_.beginGpuSection(cmd, currentFrame_);
        std::array<VkClearValue, 4> clears{};
        clears[3].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo sceneRp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        sceneRp.renderPass = sceneRP_;
        sceneRp.framebuffer = sceneFB_;
        sceneRp.renderArea = {{0, 0}, extent_};
        sceneRp.clearValueCount = static_cast<uint32_t>(clears.size());
        sceneRp.pClearValues = clears.data();
        vkCmdBeginRenderPass(cmd, &sceneRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geomPipeline_);
        VkViewport vp{0, 0, (float)extent_.width, (float)extent_.height, 0, 1};
        VkRect2D sc{{0, 0}, extent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        VkBuffer vb[] = {vertexBuffer_};
        VkDeviceSize off[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geomLayout_, 0, 1, &geomSets_[currentFrame_], 0, nullptr);
        vkCmdDraw(cmd, static_cast<uint32_t>(SCENE.size()), 1, 0, 0);
        vkCmdEndRenderPass(cmd);

        VkClearValue reflClear{};
        VkRenderPassBeginInfo ssrRp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        ssrRp.renderPass = ssrRP_;
        ssrRp.framebuffer = ssrFB_;
        ssrRp.renderArea = {{0, 0}, extent_};
        ssrRp.clearValueCount = 1;
        ssrRp.pClearValues = &reflClear;
        vkCmdBeginRenderPass(cmd, &ssrRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssrPipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssrLayout_, 0, 1, &ssrSets_[currentFrame_], 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);

        VkClearValue compClear{};
        compClear.color = {{0.05f, 0.06f, 0.1f, 1.0f}};
        VkRenderPassBeginInfo compRp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        compRp.renderPass = compositeRP_;
        compRp.framebuffer = compositeFBs_[imageIndex];
        compRp.renderArea = {{0, 0}, extent_};
        compRp.clearValueCount = 1;
        compRp.pClearValues = &compClear;
        vkCmdBeginRenderPass(cmd, &compRp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositeLayout_, 0, 1, &compositeSets_[currentFrame_], 0, nullptr);
        const float mixStrength = 0.65f;
        vkCmdPushConstants(cmd, compositeLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &mixStrength);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);
        interactive_.endGpuSection(cmd, currentFrame_);
        vkEndCommandBuffer(cmd);
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlight_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t img = 0;
        if (vkAcquireNextImageKHR(
                device_, swapchain_, UINT64_MAX, imageAvailable_[currentFrame_], VK_NULL_HANDLE, &img) ==
            VK_ERROR_OUT_OF_DATE_KHR) {
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
        if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        }
        interactive_.endFrame(currentFrame_);
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void mainLoop() {
        std::cout << "🪞 SSR 运行中（ESC 退出，屏幕外反射使用 Cubemap）\n";
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
        ci.presentMode = chooseSwapPresentMode(sc.presentModes);
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

    void createCommandPool() {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
    }

    void createVertexBuffer() {
        const VkDeviceSize sz = sizeof(GVertex) * SCENE.size();
        createBuffer(physicalDevice_,
                     device_,
                     sz,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_,
                     vertexMem_);
        void* mapped = nullptr;
        vkMapMemory(device_, vertexMem_, 0, sz, 0, &mapped);
        std::memcpy(mapped, SCENE.data(), static_cast<size_t>(sz));
        vkUnmapMemory(device_, vertexMem_);
    }

    void createUniformBuffers() {
        geomUBOs_.resize(MAX_FRAMES);
        geomUBOMem_.resize(MAX_FRAMES);
        geomMapped_.resize(MAX_FRAMES);
        ssrUBOs_.resize(MAX_FRAMES);
        ssrUBOMem_.resize(MAX_FRAMES);
        ssrMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physicalDevice_,
                         device_,
                         sizeof(GeomUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         geomUBOs_[i],
                         geomUBOMem_[i]);
            vkMapMemory(device_, geomUBOMem_[i], 0, sizeof(GeomUBO), 0, &geomMapped_[i]);
            createBuffer(physicalDevice_,
                         device_,
                         sizeof(SSRUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         ssrUBOs_[i],
                         ssrUBOMem_[i]);
            vkMapMemory(device_, ssrUBOMem_[i], 0, sizeof(SSRUBO), 0, &ssrMapped_[i]);
        }
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> sizes = {{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES * 2},
                                                      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES * 8 + 2}}};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        ci.pPoolSizes = sizes.data();
        ci.maxSets = MAX_FRAMES * 3;
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool_));
    }

    void createDescriptorSets() {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool_;
        ai.descriptorSetCount = MAX_FRAMES;
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES, geomDSL_);
        ai.pSetLayouts = layouts.data();
        geomSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, geomSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{geomUBOs_[i], 0, sizeof(GeomUBO)};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                   nullptr,
                                   geomSets_[i],
                                   0,
                                   0,
                                   1,
                                   VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                   nullptr,
                                   &bi,
                                   nullptr};
            vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
        }
        layouts.assign(MAX_FRAMES, ssrDSL_);
        ai.pSetLayouts = layouts.data();
        ssrSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, ssrSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorImageInfo color{linearSampler_, colorView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo normal{linearSampler_, normalView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo depth{linearSampler_, depthView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo cube{cubeSampler_, cubeView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorBufferInfo sbi{ssrUBOs_[i], 0, sizeof(SSRUBO)};
            std::array<VkWriteDescriptorSet, 5> ws = {{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                        nullptr,
                                                        ssrSets_[i],
                                                        0,
                                                        0,
                                                        1,
                                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                        &color,
                                                        nullptr,
                                                        nullptr},
                                                       {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                        nullptr,
                                                        ssrSets_[i],
                                                        1,
                                                        0,
                                                        1,
                                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                        &normal,
                                                        nullptr,
                                                        nullptr},
                                                       {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                        nullptr,
                                                        ssrSets_[i],
                                                        2,
                                                        0,
                                                        1,
                                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                        &depth,
                                                        nullptr,
                                                        nullptr},
                                                       {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                        nullptr,
                                                        ssrSets_[i],
                                                        3,
                                                        0,
                                                        1,
                                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                        &cube,
                                                        nullptr,
                                                        nullptr},
                                                       {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                        nullptr,
                                                        ssrSets_[i],
                                                        4,
                                                        0,
                                                        1,
                                                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                        nullptr,
                                                        &sbi,
                                                        nullptr}}};
            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(ws.size()), ws.data(), 0, nullptr);
        }
        layouts.assign(MAX_FRAMES, compositeDSL_);
        ai.pSetLayouts = layouts.data();
        compositeSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, compositeSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorImageInfo color{linearSampler_, colorView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo refl{linearSampler_, reflView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::array<VkWriteDescriptorSet, 2> ws = {{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                        nullptr,
                                                        compositeSets_[i],
                                                        0,
                                                        0,
                                                        1,
                                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                        &color,
                                                        nullptr,
                                                        nullptr},
                                                       {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                        nullptr,
                                                        compositeSets_[i],
                                                        1,
                                                        0,
                                                        1,
                                                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                        &refl,
                                                        nullptr,
                                                        nullptr}}};
            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(ws.size()), ws.data(), 0, nullptr);
        }
    }

    void createCommandBuffers() {
        cmdBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = MAX_FRAMES;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, cmdBuffers_.data()));
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

    void destroySceneTargets() {
        auto destroy = [&](VkImage& i, VkDeviceMemory& m, VkImageView& v) {
            vkDestroyImageView(device_, v, nullptr);
            vkDestroyImage(device_, i, nullptr);
            vkFreeMemory(device_, m, nullptr);
        };
        destroy(posImage_, posMem_, posView_);
        destroy(normalImage_, normalMem_, normalView_);
        destroy(colorImage_, colorMem_, colorView_);
        destroy(depthImage_, depthMem_, depthView_);
        destroy(reflImage_, reflMem_, reflView_);
    }

    void recreateSwapchain() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        vkDestroyFramebuffer(device_, sceneFB_, nullptr);
        vkDestroyFramebuffer(device_, ssrFB_, nullptr);
        for (auto fb : compositeFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        destroySceneTargets();
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        createSceneTargets();
        createReflectionTarget();
        createFramebuffers();
        createDescriptorSets();
        interactive_.onSwapchainRecreated(compositeRP_, swapFormat_, static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup() {
        vkDestroySampler(device_, linearSampler_, nullptr);
        vkDestroySampler(device_, cubeSampler_, nullptr);
        vkDestroyImageView(device_, cubeView_, nullptr);
        vkDestroyImage(device_, cubeImage_, nullptr);
        vkFreeMemory(device_, cubeMem_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, geomUBOs_[i], nullptr);
            vkFreeMemory(device_, geomUBOMem_[i], nullptr);
            vkDestroyBuffer(device_, ssrUBOs_[i], nullptr);
            vkFreeMemory(device_, ssrUBOMem_[i], nullptr);
        }
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, geomDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, ssrDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, compositeDSL_, nullptr);
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMem_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailable_[i], nullptr);
            vkDestroySemaphore(device_, renderFinished_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        vkDestroyFramebuffer(device_, sceneFB_, nullptr);
        vkDestroyFramebuffer(device_, ssrFB_, nullptr);
        for (auto fb : compositeFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        destroySceneTargets();
        vkDestroyPipeline(device_, geomPipeline_, nullptr);
        vkDestroyPipeline(device_, ssrPipeline_, nullptr);
        vkDestroyPipeline(device_, compositePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, geomLayout_, nullptr);
        vkDestroyPipelineLayout(device_, ssrLayout_, nullptr);
        vkDestroyPipelineLayout(device_, compositeLayout_, nullptr);
        vkDestroyRenderPass(device_, sceneRP_, nullptr);
        vkDestroyRenderPass(device_, ssrRP_, nullptr);
        vkDestroyRenderPass(device_, compositeRP_, nullptr);
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
    std::cout << " 第47章：屏幕空间反射（SSR + Cubemap 降级）\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    Ch47App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
    return 0;
}
