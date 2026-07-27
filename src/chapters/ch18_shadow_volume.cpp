/**
 * @file ch18_shadow_volume.cpp
 * @brief Point-light shadow volumes using CPU silhouettes and stencil z-fail.
 */
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan_tutorial/features.hpp>
#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
using namespace vulkan_tutorial;
namespace {
constexpr uint32_t WINDOW_WIDTH = 1100;
constexpr uint32_t WINDOW_HEIGHT = 720;
constexpr uint32_t MAX_FRAMES = 2;
constexpr float EXTRUSION_DISTANCE = 80.0f;
constexpr VkDeviceSize MAX_VOLUME_VERTICES = 500000;
struct SceneVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};
struct Mesh {
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
};
struct Caster {
    Mesh mesh;
    glm::mat4 transform{1.0f};
};
struct SceneUBO {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::vec4 lightPosition;
    alignas(16) glm::vec4 cameraPosition;
};
struct LightingPushConstants {
    uint32_t mode;
};
struct DepthStencilResource {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};
struct DynamicVolumeBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    uint32_t vertexCount = 0;
};
struct EdgeKey {
    uint32_t first;
    uint32_t second;
    bool operator==(const EdgeKey& other) const { return first == other.first && second == other.second; }
};
struct EdgeKeyHash {
    size_t operator()(const EdgeKey& key) const {
        return (static_cast<size_t>(key.first) << 32U) ^ static_cast<size_t>(key.second);
    }
};
Mesh makeCubeMesh() {
    Mesh mesh{};
    mesh.positions = {{-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f},
                      {-0.5f, 0.5f, -0.5f},  {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f},
                      {0.5f, 0.5f, 0.5f},    {-0.5f, 0.5f, 0.5f}};
    mesh.indices = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3,
                    1, 2, 6, 1, 6, 5, 3, 7, 6, 3, 6, 2, 0, 1, 5, 0, 5, 4};
    return mesh;
}
Mesh makeCylinderMesh(uint32_t segments) {
    Mesh mesh{};
    mesh.positions.push_back({0.0f, -0.5f, 0.0f});
    mesh.positions.push_back({0.0f, 0.5f, 0.0f});
    for (uint32_t i = 0; i < segments; ++i) {
        const float angle = glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(segments);
        mesh.positions.push_back({0.5f * std::cos(angle), -0.5f, 0.5f * std::sin(angle)});
        mesh.positions.push_back({0.5f * std::cos(angle), 0.5f, 0.5f * std::sin(angle)});
    }
    for (uint32_t i = 0; i < segments; ++i) {
        const uint32_t next = (i + 1) % segments;
        const uint32_t bottom = 2 + i * 2;
        const uint32_t top = bottom + 1;
        const uint32_t nextBottom = 2 + next * 2;
        const uint32_t nextTop = nextBottom + 1;
        mesh.indices.insert(mesh.indices.end(), {0, nextBottom, bottom, 1, top, nextTop,
                                                 bottom, nextBottom, nextTop, bottom, nextTop, top});
    }
    return mesh;
}
Mesh makeSphereMesh(uint32_t subdivisions) {
    Mesh mesh{};
    mesh.positions = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    mesh.indices = {0, 2, 4, 4, 2, 1, 1, 2, 5, 5, 2, 0, 4, 3, 0, 1, 3, 4, 5, 3, 1, 0, 3, 5};
    for (uint32_t level = 0; level < subdivisions; ++level) {
        std::unordered_map<uint64_t, uint32_t> midpointCache;
        std::vector<uint32_t> refined;
        auto midpoint = [&](uint32_t a, uint32_t b) {
            const uint32_t low = std::min(a, b);
            const uint32_t high = std::max(a, b);
            const uint64_t key = (static_cast<uint64_t>(low) << 32U) | high;
            const auto found = midpointCache.find(key);
            if (found != midpointCache.end()) return found->second;
            mesh.positions.push_back(glm::normalize(mesh.positions[a] + mesh.positions[b]));
            const uint32_t index = static_cast<uint32_t>(mesh.positions.size() - 1);
            midpointCache.emplace(key, index);
            return index;
        };
        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            const uint32_t a = mesh.indices[i];
            const uint32_t b = mesh.indices[i + 1];
            const uint32_t c = mesh.indices[i + 2];
            const uint32_t ab = midpoint(a, b);
            const uint32_t bc = midpoint(b, c);
            const uint32_t ca = midpoint(c, a);
            refined.insert(refined.end(), {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca});
        }
        mesh.indices = std::move(refined);
    }
    return mesh;
}
glm::mat4 makeTransform(const glm::vec3& position, const glm::vec3& scale, float yaw = 0.0f) {
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), position);
    transform = glm::rotate(transform, yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    return glm::scale(transform, scale);
}
void appendSceneMesh(std::vector<SceneVertex>& output, const Mesh& mesh, const glm::mat4& transform,
                     const glm::vec3& color) {
    const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(transform)));
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        const glm::vec3 a = glm::vec3(transform * glm::vec4(mesh.positions[mesh.indices[i]], 1.0f));
        const glm::vec3 b = glm::vec3(transform * glm::vec4(mesh.positions[mesh.indices[i + 1]], 1.0f));
        const glm::vec3 c = glm::vec3(transform * glm::vec4(mesh.positions[mesh.indices[i + 2]], 1.0f));
        const glm::vec3 fallbackNormal = glm::normalize(glm::cross(b - a, c - a));
        for (uint32_t corner = 0; corner < 3; ++corner) {
            const uint32_t index = mesh.indices[i + corner];
            glm::vec3 normal = fallbackNormal;
            if (mesh.positions.size() > 8 && glm::length(mesh.positions[index]) > 0.1f)
                normal = glm::normalize(normalMatrix * glm::normalize(mesh.positions[index]));
            const glm::vec3 position = glm::vec3(transform * glm::vec4(mesh.positions[index], 1.0f));
            output.push_back({position, normal, color});
        }
    }
}
glm::vec3 extrudePoint(const glm::vec3& point, const glm::vec3& lightPosition) {
    return point + glm::normalize(point - lightPosition) * EXTRUSION_DISTANCE;
}
void appendShadowVolume(std::vector<glm::vec3>& output, const Caster& caster, const glm::vec3& lightPosition) {
    const Mesh& mesh = caster.mesh;
    const size_t triangleCount = mesh.indices.size() / 3;
    std::vector<glm::vec3> world(mesh.positions.size());
    std::transform(mesh.positions.begin(), mesh.positions.end(), world.begin(), [&](const glm::vec3& position) {
        return glm::vec3(caster.transform * glm::vec4(position, 1.0f));
    });
    std::vector<bool> facesLight(triangleCount, false);
    for (size_t triangle = 0; triangle < triangleCount; ++triangle) {
        const glm::vec3& a = world[mesh.indices[triangle * 3]];
        const glm::vec3& b = world[mesh.indices[triangle * 3 + 1]];
        const glm::vec3& c = world[mesh.indices[triangle * 3 + 2]];
        facesLight[triangle] = glm::dot(glm::cross(b - a, c - a), lightPosition - a) > 0.0f;
        if (!facesLight[triangle]) continue;
        const glm::vec3 farA = extrudePoint(a, lightPosition);
        const glm::vec3 farB = extrudePoint(b, lightPosition);
        const glm::vec3 farC = extrudePoint(c, lightPosition);
        output.insert(output.end(), {a, b, c, farC, farB, farA});
    }
    struct EdgeOwner { uint32_t triangle; uint32_t from; uint32_t to; };
    std::unordered_map<EdgeKey, std::vector<EdgeOwner>, EdgeKeyHash> edges;
    for (uint32_t triangle = 0; triangle < triangleCount; ++triangle) {
        for (uint32_t edge = 0; edge < 3; ++edge) {
            const uint32_t from = mesh.indices[triangle * 3 + edge];
            const uint32_t to = mesh.indices[triangle * 3 + (edge + 1) % 3];
            edges[{std::min(from, to), std::max(from, to)}].push_back({triangle, from, to});
        }
    }
    for (const auto& [key, owners] : edges) {
        (void)key;
        const bool firstFacing = facesLight[owners[0].triangle];
        const bool secondFacing = owners.size() > 1 ? facesLight[owners[1].triangle] : !firstFacing;
        if (firstFacing == secondFacing) continue;
        const EdgeOwner& owner = firstFacing ? owners[0] : owners[1];
        const glm::vec3 a = world[owner.from];
        const glm::vec3 b = world[owner.to];
        const glm::vec3 farA = extrudePoint(a, lightPosition);
        const glm::vec3 farB = extrudePoint(b, lightPosition);
        output.insert(output.end(), {a, farA, farB, a, farB, b});
    }
}
} // namespace
/** Point-light shadow volume teaching application. */
class ShadowVolumeApp {
  public:
    /** Runs the application until the window closes. */
    void run() {
        initializeWindow();
        initializeVulkan();
        executeMainLoop();
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
    QueueFamilyIndices queueIndices_{};
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchainFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainViews_;
    std::vector<DepthStencilResource> depthResources_;
    std::vector<VkFramebuffer> framebuffers_;
    VkFormat depthStencilFormat_ = VK_FORMAT_UNDEFINED;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;
    VkPipelineLayout scenePipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout volumePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline ambientPipeline_ = VK_NULL_HANDLE;
    VkPipeline volumePipeline_ = VK_NULL_HANDLE;
    VkPipeline lightPipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkBuffer sceneBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory sceneMemory_ = VK_NULL_HANDLE;
    uint32_t sceneVertexCount_ = 0;
    std::array<DynamicVolumeBuffer, MAX_FRAMES> volumeBuffers_{};
    std::array<VkBuffer, MAX_FRAMES> uniformBuffers_{};
    std::array<VkDeviceMemory, MAX_FRAMES> uniformMemories_{};
    std::array<void*, MAX_FRAMES> uniformMapped_{};
    std::array<VkSemaphore, MAX_FRAMES> imageAvailable_{};
    std::vector<VkSemaphore> renderFinished_;
    std::array<VkFence, MAX_FRAMES> inFlight_{};
    std::vector<VkFence> imageFences_;
    std::vector<SceneVertex> sceneVertices_;
    std::vector<Caster> casters_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;
    std::chrono::steady_clock::time_point startTime_ = std::chrono::steady_clock::now();
    void initializeWindow() {
        if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Shadow Volumes - Stencil Z-Fail", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* window, int, int) {
            static_cast<ShadowVolumeApp*>(glfwGetWindowUserPointer(window))->resized_ = true;
        });
    }
    void initializeVulkan() {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physicalDevice_);
        createLogicalDevice(physicalDevice_, surface_, device_, graphicsQueue_, presentQueue_, queueIndices_);
        selectDepthStencilFormat();
        createSwapchain();
        createSwapchainViews();
        createRenderPass();
        createDescriptorSetLayout();
        createPipelineLayouts();
        createPipelines();
        createCommandPool();
        buildScene();
        createSceneBuffer();
        createPerFrameBuffers();
        createDescriptorPoolAndSets();
        createDepthResourcesAndFramebuffers();
        createCommandBuffers();
        createSyncObjects();
        std::cout << "Shadow Volume initialized: " << sceneVertexCount_ / 3 << " scene triangles, "
                  << casters_.size() << " closed casters\n";
    }
    void selectDepthStencilFormat() {
        const std::array<VkFormat, 2> candidates = {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
        for (VkFormat format : candidates) {
            if (isFormatSupported(physicalDevice_, format, VK_IMAGE_TILING_OPTIMAL,
                                  VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
                depthStencilFormat_ = format;
                return;
            }
        }
        throw std::runtime_error("Shadow volumes require a depth format with an 8-bit stencil component");
    }
    void createSwapchain() {
        const SwapChainSupportDetails support = querySwapChainSupport(physicalDevice_, surface_);
        const VkSurfaceFormatKHR format = chooseSwapSurfaceFormat(support.formats);
        const VkPresentModeKHR mode = chooseSwapPresentMode(support.presentModes);
        swapchainExtent_ = chooseSwapExtent(support.capabilities, window_);
        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0) imageCount = std::min(imageCount, support.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        info.surface = surface_;
        info.minImageCount = imageCount;
        info.imageFormat = format.format;
        info.imageColorSpace = format.colorSpace;
        info.imageExtent = swapchainExtent_;
        info.imageArrayLayers = 1;
        info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        const uint32_t families[] = {queueIndices_.graphicsFamily.value(), queueIndices_.presentFamily.value()};
        if (families[0] != families[1]) {
            info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            info.queueFamilyIndexCount = 2;
            info.pQueueFamilyIndices = families;
        } else info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.preTransform = support.capabilities.currentTransform;
        info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        info.presentMode = mode;
        info.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &info, nullptr, &swapchain_));
        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
        swapchainImages_.resize(imageCount);
        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchainImages_.data());
        swapchainFormat_ = format.format;
    }
    void createSwapchainViews() {
        swapchainViews_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            info.image = swapchainImages_[i];
            info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            info.format = swapchainFormat_;
            info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &info, nullptr, &swapchainViews_[i]));
        }
    }
    void createRenderPass() {
        VkAttachmentDescription color{};
        color.format = swapchainFormat_;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentDescription depth{};
        depth.format = depthStencilFormat_;
        depth.samples = VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = dependency.srcStageMask;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        const std::array<VkAttachmentDescription, 2> attachments = {color, depth};
        VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        info.attachmentCount = static_cast<uint32_t>(attachments.size());
        info.pAttachments = attachments.data();
        info.subpassCount = 1;
        info.pSubpasses = &subpass;
        info.dependencyCount = 1;
        info.pDependencies = &dependency;
        VK_CHECK(vkCreateRenderPass(device_, &info, nullptr, &renderPass_));
    }
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = 1;
        info.pBindings = &binding;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &info, nullptr, &descriptorSetLayout_));
    }
    void createPipelineLayouts() {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push.size = sizeof(LightingPushConstants);
        VkPipelineLayoutCreateInfo sceneInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        sceneInfo.setLayoutCount = 1;
        sceneInfo.pSetLayouts = &descriptorSetLayout_;
        sceneInfo.pushConstantRangeCount = 1;
        sceneInfo.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device_, &sceneInfo, nullptr, &scenePipelineLayout_));
        VkPipelineLayoutCreateInfo volumeInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        volumeInfo.setLayoutCount = 1;
        volumeInfo.pSetLayouts = &descriptorSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &volumeInfo, nullptr, &volumePipelineLayout_));
    }
    VkPipeline createPipeline(bool isVolume, VkPipelineColorBlendAttachmentState blend,
                              VkPipelineDepthStencilStateCreateInfo depthStencil) {
        VkShaderModule vertex = createShaderModuleFromFile(device_, isVolume ? "shadow_volume.vert.spv" : "shadow_volume_scene.vert.spv");
        VkShaderModule fragment = createShaderModuleFromFile(device_, isVolume ? "shadow_volume.frag.spv" : "shadow_volume_scene.frag.spv");
        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", nullptr};
        VkVertexInputBindingDescription binding{0, static_cast<uint32_t>(isVolume ? sizeof(glm::vec3) : sizeof(SceneVertex)), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 3> attributes{};
        attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, isVolume ? 0U : static_cast<uint32_t>(offsetof(SceneVertex, position))};
        attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(SceneVertex, normal))};
        attributes[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(SceneVertex, color))};
        VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount = isVolume ? 1 : static_cast<uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = isVolume ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        raster.depthClampEnable = VK_FALSE;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments = &blend;
        const std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.stageCount = static_cast<uint32_t>(stages.size());
        info.pStages = stages.data();
        info.pVertexInputState = &vertexInput;
        info.pInputAssemblyState = &assembly;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depthStencil;
        info.pColorBlendState = &colorBlend;
        info.pDynamicState = &dynamic;
        info.layout = isVolume ? volumePipelineLayout_ : scenePipelineLayout_;
        info.renderPass = renderPass_;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
        vkDestroyShaderModule(device_, fragment, nullptr);
        vkDestroyShaderModule(device_, vertex, nullptr);
        return pipeline;
    }
    void createPipelines() {
        VkPipelineColorBlendAttachmentState opaque{};
        opaque.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineDepthStencilStateCreateInfo ambientDepth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ambientDepth.depthTestEnable = VK_TRUE;
        ambientDepth.depthWriteEnable = VK_TRUE;
        ambientDepth.depthCompareOp = VK_COMPARE_OP_LESS;
        ambientPipeline_ = createPipeline(false, opaque, ambientDepth);
        VkPipelineColorBlendAttachmentState noColor{};
        noColor.colorWriteMask = 0;
        VkStencilOpState front{};
        front.failOp = VK_STENCIL_OP_KEEP;
        front.passOp = VK_STENCIL_OP_KEEP;
        front.depthFailOp = VK_STENCIL_OP_DECREMENT_AND_WRAP;
        front.compareOp = VK_COMPARE_OP_ALWAYS;
        front.compareMask = 0xFF;
        front.writeMask = 0xFF;
        VkStencilOpState back = front;
        back.depthFailOp = VK_STENCIL_OP_INCREMENT_AND_WRAP;
        VkPipelineDepthStencilStateCreateInfo volumeDepth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        volumeDepth.depthTestEnable = VK_TRUE;
        volumeDepth.depthWriteEnable = VK_FALSE;
        volumeDepth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        volumeDepth.stencilTestEnable = VK_TRUE;
        volumeDepth.front = front;
        volumeDepth.back = back;
        volumePipeline_ = createPipeline(true, noColor, volumeDepth);
        VkPipelineColorBlendAttachmentState additive = opaque;
        additive.blendEnable = VK_TRUE;
        additive.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        additive.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        additive.colorBlendOp = VK_BLEND_OP_ADD;
        additive.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        additive.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        additive.alphaBlendOp = VK_BLEND_OP_ADD;
        VkStencilOpState litStencil{};
        litStencil.failOp = VK_STENCIL_OP_KEEP;
        litStencil.passOp = VK_STENCIL_OP_KEEP;
        litStencil.depthFailOp = VK_STENCIL_OP_KEEP;
        litStencil.compareOp = VK_COMPARE_OP_EQUAL;
        litStencil.compareMask = 0xFF;
        litStencil.writeMask = 0;
        litStencil.reference = 0;
        VkPipelineDepthStencilStateCreateInfo lightDepth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        lightDepth.depthTestEnable = VK_TRUE;
        lightDepth.depthWriteEnable = VK_FALSE;
        lightDepth.depthCompareOp = VK_COMPARE_OP_EQUAL;
        lightDepth.stencilTestEnable = VK_TRUE;
        lightDepth.front = litStencil;
        lightDepth.back = litStencil;
        lightPipeline_ = createPipeline(false, additive, lightDepth);
    }
    void createCommandPool() {
        VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &info, nullptr, &commandPool_));
    }
    void addObject(const Mesh& mesh, const glm::mat4& transform, const glm::vec3& color, bool castsShadow) {
        appendSceneMesh(sceneVertices_, mesh, transform, color);
        if (castsShadow) casters_.push_back({mesh, transform});
    }
    void buildScene() {
        const Mesh cube = makeCubeMesh();
        const Mesh cylinder = makeCylinderMesh(20);
        const Mesh sphere = makeSphereMesh(2);
        addObject(cube, makeTransform({0.0f, -0.65f, 0.0f}, {12.0f, 0.3f, 12.0f}), {0.30f, 0.34f, 0.39f}, false);
        addObject(cube, makeTransform({0.0f, 2.8f, -6.0f}, {12.0f, 7.0f, 0.25f}), {0.22f, 0.26f, 0.31f}, false);
        addObject(cube, makeTransform({-6.0f, 2.1f, 0.0f}, {0.25f, 5.5f, 12.0f}), {0.26f, 0.29f, 0.34f}, false);
        addObject(cube, makeTransform({0.0f, 0.2f, 0.0f}, {2.4f, 1.4f, 2.4f}, 0.35f), {0.72f, 0.24f, 0.18f}, true);
        addObject(cylinder, makeTransform({-3.4f, 1.0f, -1.8f}, {1.2f, 3.3f, 1.2f}), {0.18f, 0.55f, 0.70f}, true);
        addObject(cylinder, makeTransform({3.6f, 0.6f, -2.8f}, {0.9f, 2.5f, 0.9f}), {0.75f, 0.58f, 0.16f}, true);
        addObject(sphere, makeTransform({-2.8f, 0.45f, 2.4f}, {1.25f, 1.25f, 1.25f}), {0.25f, 0.68f, 0.30f}, true);
        addObject(sphere, makeTransform({3.2f, 1.2f, 2.6f}, {1.7f, 1.7f, 1.7f}), {0.55f, 0.24f, 0.68f}, true);
        for (int i = 0; i < 7; ++i) {
            const float angle = static_cast<float>(i) * 0.82f;
            const glm::vec3 position{4.5f * std::cos(angle), -0.15f + 0.22f * (i % 3), 4.5f * std::sin(angle)};
            const glm::vec3 scale{0.55f + 0.12f * (i % 2), 0.8f + 0.25f * (i % 3), 0.55f + 0.12f * (i % 2)};
            const glm::vec3 color{0.25f + 0.08f * i, 0.48f - 0.035f * i, 0.65f - 0.04f * i};
            addObject(cube, makeTransform(position, scale, angle * 0.7f), color, true);
        }
    }
    void uploadStaticBuffer(const void* data, VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory) {
        createBuffer(physicalDevice_, device_, size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
        void* mapped = nullptr;
        VK_CHECK(vkMapMemory(device_, memory, 0, size, 0, &mapped));
        std::memcpy(mapped, data, static_cast<size_t>(size));
        vkUnmapMemory(device_, memory);
    }
    void createSceneBuffer() {
        sceneVertexCount_ = static_cast<uint32_t>(sceneVertices_.size());
        uploadStaticBuffer(sceneVertices_.data(), sceneVertices_.size() * sizeof(SceneVertex), sceneBuffer_, sceneMemory_);
    }
    void createPerFrameBuffers() {
        for (uint32_t frame = 0; frame < MAX_FRAMES; ++frame) {
            createBuffer(physicalDevice_, device_, sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers_[frame], uniformMemories_[frame]);
            VK_CHECK(vkMapMemory(device_, uniformMemories_[frame], 0, sizeof(SceneUBO), 0, &uniformMapped_[frame]));
            const VkDeviceSize volumeSize = MAX_VOLUME_VERTICES * sizeof(glm::vec3);
            createBuffer(physicalDevice_, device_, volumeSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         volumeBuffers_[frame].buffer, volumeBuffers_[frame].memory);
            VK_CHECK(vkMapMemory(device_, volumeBuffers_[frame].memory, 0, volumeSize, 0, &volumeBuffers_[frame].mapped));
        }
    }
    void createDescriptorPoolAndSets() {
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = MAX_FRAMES;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &size;
        VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));
        const std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES, descriptorSetLayout_);
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = descriptorPool_;
        allocate.descriptorSetCount = MAX_FRAMES;
        allocate.pSetLayouts = layouts.data();
        descriptorSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &allocate, descriptorSets_.data()));
        for (uint32_t frame = 0; frame < MAX_FRAMES; ++frame) {
            VkDescriptorBufferInfo bufferInfo{uniformBuffers_[frame], 0, sizeof(SceneUBO)};
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = descriptorSets_[frame];
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.pBufferInfo = &bufferInfo;
            vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        }
    }
    void createDepthResourcesAndFramebuffers() {
        depthResources_.resize(swapchainImages_.size());
        framebuffers_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            DepthStencilResource& depth = depthResources_[i];
            createImage(physicalDevice_, device_, swapchainExtent_.width, swapchainExtent_.height, depthStencilFormat_,
                        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depth.image, depth.memory);
            VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image = depth.image;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = depthStencilFormat_;
            view.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &view, nullptr, &depth.view));
            const std::array<VkImageView, 2> attachments = {swapchainViews_[i], depth.view};
            VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            framebuffer.renderPass = renderPass_;
            framebuffer.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebuffer.pAttachments = attachments.data();
            framebuffer.width = swapchainExtent_.width;
            framebuffer.height = swapchainExtent_.height;
            framebuffer.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &framebuffer, nullptr, &framebuffers_[i]));
        }
        imageFences_.assign(swapchainImages_.size(), VK_NULL_HANDLE);
    }
    void createCommandBuffers() {
        commandBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        info.commandPool = commandPool_;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = MAX_FRAMES;
        VK_CHECK(vkAllocateCommandBuffers(device_, &info, commandBuffers_.data()));
    }
    void createSyncObjects() {
        renderFinished_.resize(swapchainImages_.size());
        VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (uint32_t frame = 0; frame < MAX_FRAMES; ++frame) {
            VK_CHECK(vkCreateSemaphore(device_, &semaphore, nullptr, &imageAvailable_[frame]));
            VK_CHECK(vkCreateFence(device_, &fence, nullptr, &inFlight_[frame]));
        }
        for (VkSemaphore& signalSemaphore : renderFinished_)
            VK_CHECK(vkCreateSemaphore(device_, &semaphore, nullptr, &signalSemaphore));
    }
    SceneUBO updateFrameData(uint32_t frame) {
        const float time = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime_).count();
        const glm::vec3 cameraPosition{11.0f * std::cos(time * 0.10f), 7.0f, 11.0f * std::sin(time * 0.10f)};
        const glm::vec3 lightPosition{4.5f * std::cos(time * 0.55f), 5.0f + std::sin(time * 0.8f), 4.5f * std::sin(time * 0.55f)};
        SceneUBO ubo{};
        ubo.view = glm::lookAt(cameraPosition, glm::vec3(0.0f, 0.7f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.projection = glm::perspective(glm::radians(52.0f), static_cast<float>(swapchainExtent_.width) /
                                         static_cast<float>(swapchainExtent_.height), 0.1f, 120.0f);
        ubo.projection[1][1] *= -1.0f;
        ubo.lightPosition = glm::vec4(lightPosition, 1.0f);
        ubo.cameraPosition = glm::vec4(cameraPosition, 1.0f);
        std::memcpy(uniformMapped_[frame], &ubo, sizeof(ubo));
        std::vector<glm::vec3> volumeVertices;
        volumeVertices.reserve(120000);
        for (const Caster& caster : casters_) appendShadowVolume(volumeVertices, caster, lightPosition);
        if (volumeVertices.size() > MAX_VOLUME_VERTICES) throw std::runtime_error("Shadow volume vertex capacity exceeded");
        std::memcpy(volumeBuffers_[frame].mapped, volumeVertices.data(), volumeVertices.size() * sizeof(glm::vec3));
        volumeBuffers_[frame].vertexCount = static_cast<uint32_t>(volumeVertices.size());
        return ubo;
    }
    void bindCommonState(VkCommandBuffer command) {
        const VkViewport viewport{0.0f, 0.0f, static_cast<float>(swapchainExtent_.width),
                                  static_cast<float>(swapchainExtent_.height), 0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, swapchainExtent_};
        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &scissor);
    }
    void recordCommandBuffer(VkCommandBuffer command, uint32_t imageIndex) {
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(command, &begin));
        std::array<VkClearValue, 2> clears{};
        clears[0].color = {{0.012f, 0.016f, 0.025f, 1.0f}};
        clears[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo renderPass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        renderPass.renderPass = renderPass_;
        renderPass.framebuffer = framebuffers_[imageIndex];
        renderPass.renderArea = {{0, 0}, swapchainExtent_};
        renderPass.clearValueCount = static_cast<uint32_t>(clears.size());
        renderPass.pClearValues = clears.data();
        vkCmdBeginRenderPass(command, &renderPass, VK_SUBPASS_CONTENTS_INLINE);
        bindCommonState(command);
        const VkDeviceSize offset = 0;
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout_, 0, 1,
                                &descriptorSets_[currentFrame_], 0, nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, ambientPipeline_);
        const LightingPushConstants ambient{0};
        vkCmdPushConstants(command, scenePipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ambient), &ambient);
        vkCmdBindVertexBuffers(command, 0, 1, &sceneBuffer_, &offset);
        vkCmdDraw(command, sceneVertexCount_, 1, 0, 0);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, volumePipelineLayout_, 0, 1,
                                &descriptorSets_[currentFrame_], 0, nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, volumePipeline_);
        vkCmdBindVertexBuffers(command, 0, 1, &volumeBuffers_[currentFrame_].buffer, &offset);
        vkCmdDraw(command, volumeBuffers_[currentFrame_].vertexCount, 1, 0, 0);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout_, 0, 1,
                                &descriptorSets_[currentFrame_], 0, nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, lightPipeline_);
        const LightingPushConstants direct{1};
        vkCmdPushConstants(command, scenePipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(direct), &direct);
        vkCmdBindVertexBuffers(command, 0, 1, &sceneBuffer_, &offset);
        vkCmdDraw(command, sceneVertexCount_, 1, 0, 0);
        vkCmdEndRenderPass(command);
        VK_CHECK(vkEndCommandBuffer(command));
    }
    void drawFrame() {
        VK_CHECK(vkWaitForFences(device_, 1, &inFlight_[currentFrame_], VK_TRUE, UINT64_MAX));
        uint32_t imageIndex = 0;
        VkResult result = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailable_[currentFrame_],
                                                VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) throw std::runtime_error("Failed to acquire swapchain image");
        if (imageFences_[imageIndex] != VK_NULL_HANDLE)
            VK_CHECK(vkWaitForFences(device_, 1, &imageFences_[imageIndex], VK_TRUE, UINT64_MAX));
        imageFences_[imageIndex] = inFlight_[currentFrame_];
        updateFrameData(currentFrame_);
        VK_CHECK(vkResetFences(device_, 1, &inFlight_[currentFrame_]));
        VK_CHECK(vkResetCommandBuffer(commandBuffers_[currentFrame_], 0));
        recordCommandBuffer(commandBuffers_[currentFrame_], imageIndex);
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &imageAvailable_[currentFrame_];
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffers_[currentFrame_];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished_[imageIndex];
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &submit, inFlight_[currentFrame_]));
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &renderFinished_[imageIndex];
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain_;
        present.pImageIndices = &imageIndex;
        result = vkQueuePresentKHR(presentQueue_, &present);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        } else if (result != VK_SUCCESS) throw std::runtime_error("Failed to present swapchain image");
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }
    void destroySwapchainResources() {
        for (VkFramebuffer framebuffer : framebuffers_) vkDestroyFramebuffer(device_, framebuffer, nullptr);
        framebuffers_.clear();
        for (DepthStencilResource& depth : depthResources_) {
            vkDestroyImageView(device_, depth.view, nullptr);
            vkDestroyImage(device_, depth.image, nullptr);
            vkFreeMemory(device_, depth.memory, nullptr);
        }
        depthResources_.clear();
        for (VkImageView view : swapchainViews_) vkDestroyImageView(device_, view, nullptr);
        swapchainViews_.clear();
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
    void recreateSwapchain() {
        int width = 0;
        int height = 0;
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window_, &width, &height);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        destroySwapchainResources();
        createSwapchain();
        createSwapchainViews();
        createDepthResourcesAndFramebuffers();
    }
    void executeMainLoop() {
        std::cout << "Running stencil z-fail shadow volumes. Press Escape to exit.\n";
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }
    void cleanup() {
        destroySwapchainResources();
        for (uint32_t frame = 0; frame < MAX_FRAMES; ++frame) {
            vkDestroySemaphore(device_, imageAvailable_[frame], nullptr);
        for (VkSemaphore semaphore : renderFinished_) vkDestroySemaphore(device_, semaphore, nullptr);
            vkDestroyFence(device_, inFlight_[frame], nullptr);
            vkUnmapMemory(device_, uniformMemories_[frame]);
            vkDestroyBuffer(device_, uniformBuffers_[frame], nullptr);
            vkFreeMemory(device_, uniformMemories_[frame], nullptr);
            vkUnmapMemory(device_, volumeBuffers_[frame].memory);
            vkDestroyBuffer(device_, volumeBuffers_[frame].buffer, nullptr);
            vkFreeMemory(device_, volumeBuffers_[frame].memory, nullptr);
        }
        vkDestroyBuffer(device_, sceneBuffer_, nullptr);
        vkFreeMemory(device_, sceneMemory_, nullptr);
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        vkDestroyPipeline(device_, lightPipeline_, nullptr);
        vkDestroyPipeline(device_, volumePipeline_, nullptr);
        vkDestroyPipeline(device_, ambientPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, volumePipelineLayout_, nullptr);
        vkDestroyPipelineLayout(device_, scenePipelineLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
};
int main() {
    ShadowVolumeApp application;
    try {
        application.run();
    } catch (const std::exception& exception) {
        std::cerr << "Shadow Volume error: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}
