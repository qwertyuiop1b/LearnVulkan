/**
 * @file ch59_reflection_probe.cpp
 * @brief 第59章：反射探针（Reflection Probe / Cubemap 环境反射）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【原理】
 *  反射探针在场景中的某个点捕获周围环境，存入 Cubemap。
 *  反射材质采样 Cubemap 实现环境反射。
 *
 * 【Cubemap 捕获流程】
 *  - 创建 VkImage（flags=VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT，arrayLayers=6）
 *  - 向 6 个方向（+X -X +Y -Y +Z -Z）各渲染一次，90° FOV
 *  - 每次渲染写入 Cubemap 的一个 face
 *  - 使用 VkImageView（viewType=VK_IMAGE_VIEW_TYPE_CUBE）采样
 *
 * 【反射着色器】probe_scene.vert/frag
 *  - 漫反射 + Fresnel 控制的 Cubemap 反射
 *  - ImGui 调节 metallic、roughness
 *
 * 【捕获时机】
 *  本章在初始化时一次性捕获（离线探针），不每帧重新捕获。
 *  按 R 键可重新捕获（相机移动后场景视角变化）。
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>
#include <vulkan_tutorial/interactive_chapter.hpp>

#include <imgui.h>

#include <array>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH      = 800;
constexpr uint32_t HEIGHT     = 600;
constexpr int      MAX_FRAMES = 2;
constexpr uint32_t PROBE_SIZE = 256;   // Cubemap 每面分辨率

struct SceneVertex { glm::vec3 pos; glm::vec3 normal; glm::vec3 color; };
struct CapturePC   { alignas(16) glm::mat4 viewProj; };

struct ProbeSceneUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 cameraPos;
    float roughness;
    float metallic;
    float pad[2];
};

// 场景：地面 + 4 根彩色柱 + 中心金属球（用多面体近似）
static const std::vector<SceneVertex> ENV_SCENE = {
    // 地面
    {{-6,-1,-6},{0,1,0},{0.4f,0.38f,0.3f}}, {{ 6,-1,-6},{0,1,0},{0.4f,0.38f,0.3f}},
    {{ 6,-1, 6},{0,1,0},{0.4f,0.38f,0.3f}}, {{-6,-1,-6},{0,1,0},{0.4f,0.38f,0.3f}},
    {{ 6,-1, 6},{0,1,0},{0.4f,0.38f,0.3f}}, {{-6,-1, 6},{0,1,0},{0.4f,0.38f,0.3f}},
    // 天空面片（蓝天）
    {{-10,5,-10},{0,-1,0},{0.4f,0.6f,1.0f}}, {{ 10,5,-10},{0,-1,0},{0.4f,0.6f,1.0f}},
    {{ 10,5, 10},{0,-1,0},{0.4f,0.6f,1.0f}}, {{-10,5,-10},{0,-1,0},{0.4f,0.6f,1.0f}},
    {{ 10,5, 10},{0,-1,0},{0.4f,0.6f,1.0f}}, {{-10,5, 10},{0,-1,0},{0.4f,0.6f,1.0f}},
    // 红色柱（+X）
    {{ 3,-1,-0.3f},{-1,0,0},{1,0.2f,0.2f}}, {{ 3,-1, 0.3f},{-1,0,0},{1,0.2f,0.2f}},
    {{ 3, 3, 0.3f},{-1,0,0},{1,0.2f,0.2f}}, {{ 3,-1,-0.3f},{-1,0,0},{1,0.2f,0.2f}},
    {{ 3, 3, 0.3f},{-1,0,0},{1,0.2f,0.2f}}, {{ 3, 3,-0.3f},{-1,0,0},{1,0.2f,0.2f}},
    // 蓝色柱（-X）
    {{-3,-1,-0.3f},{1,0,0},{0.2f,0.4f,1}}, {{-3,-1, 0.3f},{1,0,0},{0.2f,0.4f,1}},
    {{-3, 3, 0.3f},{1,0,0},{0.2f,0.4f,1}}, {{-3,-1,-0.3f},{1,0,0},{0.2f,0.4f,1}},
    {{-3, 3, 0.3f},{1,0,0},{0.2f,0.4f,1}}, {{-3, 3,-0.3f},{1,0,0},{0.2f,0.4f,1}},
    // 绿色柱（+Z）
    {{-0.3f,-1, 3},{0,0,-1},{0.2f,1,0.2f}}, {{ 0.3f,-1, 3},{0,0,-1},{0.2f,1,0.2f}},
    {{ 0.3f, 3, 3},{0,0,-1},{0.2f,1,0.2f}}, {{-0.3f,-1, 3},{0,0,-1},{0.2f,1,0.2f}},
    {{ 0.3f, 3, 3},{0,0,-1},{0.2f,1,0.2f}}, {{-0.3f, 3, 3},{0,0,-1},{0.2f,1,0.2f}},
};

// 中心反射球（粗球体近似，16 面）
static std::vector<SceneVertex> buildSphere(float r=0.8f, glm::vec3 col={0.9f,0.85f,0.7f})
{
    std::vector<SceneVertex> v;
    constexpr int N=12;
    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            auto p=[&](int ii,int jj)->glm::vec3{
                float phi=float(ii)/N*3.14159f;
                float theta=float(jj)/N*6.28318f;
                return {r*sinf(phi)*cosf(theta), r*cosf(phi), r*sinf(phi)*sinf(theta)};
            };
            glm::vec3 p00=p(i,j), p10=p(i+1,j), p01=p(i,j+1), p11=p(i+1,j+1);
            auto add=[&](glm::vec3 a,glm::vec3 b,glm::vec3 c){
                v.push_back({a,glm::normalize(a),col});
                v.push_back({b,glm::normalize(b),col});
                v.push_back({c,glm::normalize(c),col});
            };
            add(p00,p10,p11); add(p00,p11,p01);
        }
    }
    return v;
}

class Ch59App {
public:
    void run() { initWindow(); initVulkan(); mainLoop(); cleanup(); }

private:
    GLFWwindow*      window_  = nullptr;
    VkInstance       instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_  = VK_NULL_HANDLE;
    VkPhysicalDevice physDev_  = VK_NULL_HANDLE;
    VkDevice         device_   = VK_NULL_HANDLE;
    VkQueue          gQueue_   = VK_NULL_HANDLE;
    VkQueue          pQueue_   = VK_NULL_HANDLE;
    QueueFamilyIndices queueIdx_{};
    VkSwapchainKHR   swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage>       swapImages_;
    std::vector<VkImageView>   swapViews_;
    std::vector<VkFramebuffer> swapFBs_;
    VkFormat   swapFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    DepthResources screenDepth_{};
    VkCommandPool  cmdPool_  = VK_NULL_HANDLE;

    // Cubemap
    VkImage        cubeImage_ = VK_NULL_HANDLE;
    VkDeviceMemory cubeMem_   = VK_NULL_HANDLE;
    VkImageView    cubeView_  = VK_NULL_HANDLE;  // VK_IMAGE_VIEW_TYPE_CUBE（采样用）
    // 每个 face 的 ImageView（渲染用）
    std::array<VkImageView,    6> cubeFaceViews_{};
    std::array<VkFramebuffer,  6> captureFBs_{};
    VkImage        capDepthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory capDepthMem_   = VK_NULL_HANDLE;
    VkImageView    capDepthView_  = VK_NULL_HANDLE;
    VkRenderPass   captureRP_ = VK_NULL_HANDLE;

    // Scene RP
    VkRenderPass sceneRP_ = VK_NULL_HANDLE;

    // Pipelines
    VkPipeline       capturePipeline_  = VK_NULL_HANDLE;
    VkPipelineLayout captureLayout_    = VK_NULL_HANDLE;
    VkPipeline       scenePipeline_    = VK_NULL_HANDLE;
    VkPipelineLayout sceneLayout_      = VK_NULL_HANDLE;
    VkDescriptorSetLayout sceneDSL_    = VK_NULL_HANDLE;

    VkDescriptorPool             pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneSets_;
    VkSampler                    sampler_ = VK_NULL_HANDLE;

    // Geometry
    VkBuffer       envVB_    = VK_NULL_HANDLE; VkDeviceMemory envVBMem_  = VK_NULL_HANDLE;
    VkBuffer       sphereVB_ = VK_NULL_HANDLE; VkDeviceMemory sphereVBMem_ = VK_NULL_HANDLE;
    uint32_t       sphereVerts_ = 0;
    std::vector<VkBuffer>       sceneUBOs_;
    std::vector<VkDeviceMemory> sceneUBOMem_;
    std::vector<void*>          sceneMapped_;

    std::vector<VkCommandBuffer> cmds_;
    std::vector<VkSemaphore>     imgAvail_, renderDone_;
    std::vector<VkFence>         inFlight_;
    uint32_t frame_ = 0;
    bool     resized_ = false;

    InteractiveChapterTools interactive_;

    float metallic_   = 0.9f;
    float roughness_  = 0.1f;
    bool  showReflect_= true;
    bool  needRecapture_ = false;

    // Cubemap 6 面的视角矩阵（探针在原点）
    static glm::mat4 cubeFaceViewProj(int face)
    {
        glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 50.0f);
        proj[1][1] *= -1;
        std::array<glm::mat4,6> views = {{
            glm::lookAt(glm::vec3(0),{+1,0,0},{0,-1,0}),  // +X
            glm::lookAt(glm::vec3(0),{-1,0,0},{0,-1,0}),  // -X
            glm::lookAt(glm::vec3(0),{0,+1,0},{0,0,+1}),  // +Y
            glm::lookAt(glm::vec3(0),{0,-1,0},{0,0,-1}),  // -Y
            glm::lookAt(glm::vec3(0),{0,0,+1},{0,-1,0}),  // +Z
            glm::lookAt(glm::vec3(0),{0,0,-1},{0,-1,0}),  // -Z
        }};
        return proj * views[face];
    }

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第59章：反射探针", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch59App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
        interactive_.attachInput(window_);
    }

    void initVulkan()
    {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physDev_);
        createLogicalDevice(physDev_, surface_, device_, gQueue_, pQueue_, queueIdx_);
        createSwapchain(); createImageViews();
        screenDepth_ = createDepthResources(physDev_, device_, extent_);
        createCommandPool(); createSampler();
        createCubemap(); createCaptureResources();
        createRenderPasses(); createSwapFramebuffers();
        uploadGeometry(); createUniformBuffers();
        createDescriptorLayout(); createDescriptorPool(); createDescriptorSets();
        createPipelines(); createCommandBuffers(); createSyncObjects();
        captureProbe();   // 初始捕获

        InteractiveInitInfo ii{};
        ii.window = window_; ii.instance = instance_;
        ii.physicalDevice = physDev_; ii.device = device_;
        ii.graphicsQueue = gQueue_; ii.queueFamily = queueIdx_.graphicsFamily.value();
        ii.renderPass = sceneRP_; ii.swapchainFormat = swapFormat_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setDistance(5.0f);
        interactive_.camera().setAngles(30.0f, 20.0f);
    }

    void createSwapchain()
    {
        auto d=querySwapChainSupport(physDev_,surface_);
        auto f=chooseSwapSurfaceFormat(d.formats);
        auto m=chooseSwapPresentMode(d.presentModes);
        extent_=chooseSwapExtent(d.capabilities,window_); swapFormat_=f.format;
        uint32_t cnt=d.capabilities.minImageCount+1;
        if(d.capabilities.maxImageCount>0) cnt=std::min(cnt,d.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface=surface_; ci.minImageCount=cnt;
        ci.imageFormat=f.format; ci.imageColorSpace=f.colorSpace;
        ci.imageExtent=extent_; ci.imageArrayLayers=1;
        ci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        uint32_t qf[2]={queueIdx_.graphicsFamily.value(),queueIdx_.presentFamily.value()};
        if(qf[0]!=qf[1]){ci.imageSharingMode=VK_SHARING_MODE_CONCURRENT;ci.queueFamilyIndexCount=2;ci.pQueueFamilyIndices=qf;}
        else ci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform=d.capabilities.currentTransform;
        ci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; ci.presentMode=m; ci.clipped=VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_,&ci,nullptr,&swapchain_));
        uint32_t n=0; vkGetSwapchainImagesKHR(device_,swapchain_,&n,nullptr);
        swapImages_.resize(n); vkGetSwapchainImagesKHR(device_,swapchain_,&n,swapImages_.data());
    }

    void createImageViews()
    {
        swapViews_.resize(swapImages_.size());
        for(size_t i=0;i<swapImages_.size();++i){
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image=swapImages_[i]; vi.viewType=VK_IMAGE_VIEW_TYPE_2D;
            vi.format=swapFormat_; vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            VK_CHECK(vkCreateImageView(device_,&vi,nullptr,&swapViews_[i]));
        }
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex=queueIdx_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_,&ci,nullptr,&cmdPool_));
    }

    void createSampler()
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter=VK_FILTER_LINEAR; si.minFilter=VK_FILTER_LINEAR;
        si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod=1.0f;
        VK_CHECK(vkCreateSampler(device_,&si,nullptr,&sampler_));
    }

    void createCubemap()
    {
        // Cubemap image（6 layers）
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.flags       = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        ci.imageType   = VK_IMAGE_TYPE_2D;
        ci.format      = swapFormat_;
        ci.extent      = {PROBE_SIZE, PROBE_SIZE, 1};
        ci.mipLevels   = 1;
        ci.arrayLayers = 6;
        ci.samples     = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
        ci.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(device_,&ci,nullptr,&cubeImage_));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(device_,cubeImage_,&mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize=mr.size;
        ai.memoryTypeIndex=findMemoryType(physDev_,mr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device_,&ai,nullptr,&cubeMem_));
        VK_CHECK(vkBindImageMemory(device_,cubeImage_,cubeMem_,0));

        // Cubemap view（用于着色器采样）
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image       = cubeImage_;
        vi.viewType    = VK_IMAGE_VIEW_TYPE_CUBE;  // ← 关键
        vi.format      = swapFormat_;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
        VK_CHECK(vkCreateImageView(device_,&vi,nullptr,&cubeView_));

        // 6 个 face view（用于渲染每个面）
        for(int f=0;f<6;++f){
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, (uint32_t)f, 1};
            VK_CHECK(vkCreateImageView(device_,&vi,nullptr,&cubeFaceViews_[f]));
        }

        // 初始 layout → SHADER_READ_ONLY
        VkCommandBuffer cmd=beginSingleTimeCommands(device_,cmdPool_);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        b.image=cubeImage_; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,6};
        b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,0,nullptr,1,&b);
        endSingleTimeCommands(device_,cmdPool_,gQueue_,cmd);
    }

    void createCaptureResources()
    {
        VkFormat depthFmt=findDepthFormat(physDev_);
        createImage(physDev_,device_,PROBE_SIZE,PROBE_SIZE,depthFmt,VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,capDepthImage_,capDepthMem_);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image=capDepthImage_; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=depthFmt;
        vi.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};
        VK_CHECK(vkCreateImageView(device_,&vi,nullptr,&capDepthView_));
    }

    void createRenderPasses()
    {
        // Capture RP（每面一次，color→SHADER_READ_ONLY）
        {
            std::array<VkAttachmentDescription,2> a{};
            a[0].format=swapFormat_; a[0].samples=VK_SAMPLE_COUNT_1_BIT;
            a[0].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; a[0].storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            a[0].initialLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            a[0].finalLayout  =VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            a[1].format=findDepthFormat(physDev_); a[1].samples=VK_SAMPLE_COUNT_1_BIT;
            a[1].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; a[1].storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a[1].stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a[1].stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a[1].initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
            a[1].finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            VkAttachmentReference cr{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkAttachmentReference dr{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{};
            sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount=1; sub.pColorAttachments=&cr; sub.pDepthStencilAttachment=&dr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount=2; rpci.pAttachments=a.data(); rpci.subpassCount=1; rpci.pSubpasses=&sub;
            VK_CHECK(vkCreateRenderPass(device_,&rpci,nullptr,&captureRP_));

            // Capture framebuffers（每面一个）
            VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fci.renderPass=captureRP_; fci.attachmentCount=2;
            fci.width=PROBE_SIZE; fci.height=PROBE_SIZE; fci.layers=1;
            for(int f=0;f<6;++f){
                std::array<VkImageView,2> atts={cubeFaceViews_[f],capDepthView_};
                fci.pAttachments=atts.data();
                VK_CHECK(vkCreateFramebuffer(device_,&fci,nullptr,&captureFBs_[f]));
            }
        }
        // Scene RP（→ swap）
        {
            std::array<VkAttachmentDescription,2> a{};
            a[0].format=swapFormat_; a[0].samples=VK_SAMPLE_COUNT_1_BIT;
            a[0].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; a[0].storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            a[0].initialLayout=VK_IMAGE_LAYOUT_UNDEFINED; a[0].finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            a[1].format=screenDepth_.format; a[1].samples=VK_SAMPLE_COUNT_1_BIT;
            a[1].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; a[1].storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a[1].stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a[1].stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a[1].initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
            a[1].finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            VkAttachmentReference cr{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkAttachmentReference dr{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{};
            sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount=1; sub.pColorAttachments=&cr; sub.pDepthStencilAttachment=&dr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount=2; rpci.pAttachments=a.data(); rpci.subpassCount=1; rpci.pSubpasses=&sub;
            VK_CHECK(vkCreateRenderPass(device_,&rpci,nullptr,&sceneRP_));
        }
    }

    void createSwapFramebuffers()
    {
        swapFBs_.resize(swapImages_.size());
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass=sceneRP_; fci.attachmentCount=2;
        fci.width=extent_.width; fci.height=extent_.height; fci.layers=1;
        for(size_t i=0;i<swapImages_.size();++i){
            std::array<VkImageView,2> atts={swapViews_[i],screenDepth_.view};
            fci.pAttachments=atts.data();
            VK_CHECK(vkCreateFramebuffer(device_,&fci,nullptr,&swapFBs_[i]));
        }
    }

    void uploadGeometry()
    {
        auto doUpload=[&](const void* data,size_t bytes,VkBuffer& buf,VkDeviceMemory& mem){
            VkDeviceSize sz=bytes;
            VkBuffer st; VkDeviceMemory stm;
            createBuffer(physDev_,device_,sz,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,st,stm);
            void* d; vkMapMemory(device_,stm,0,sz,0,&d); std::memcpy(d,data,sz); vkUnmapMemory(device_,stm);
            createBuffer(physDev_,device_,sz,
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,buf,mem);
            copyBuffer(device_,cmdPool_,gQueue_,st,buf,sz);
            vkDestroyBuffer(device_,st,nullptr); vkFreeMemory(device_,stm,nullptr);
        };
        doUpload(ENV_SCENE.data(),sizeof(SceneVertex)*ENV_SCENE.size(),envVB_,envVBMem_);
        auto sphere=buildSphere();
        sphereVerts_=static_cast<uint32_t>(sphere.size());
        doUpload(sphere.data(),sizeof(SceneVertex)*sphere.size(),sphereVB_,sphereVBMem_);
    }

    void createUniformBuffers()
    {
        sceneUBOs_.resize(MAX_FRAMES); sceneUBOMem_.resize(MAX_FRAMES); sceneMapped_.resize(MAX_FRAMES);
        for(int i=0;i<MAX_FRAMES;++i){
            createBuffer(physDev_,device_,sizeof(ProbeSceneUBO),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         sceneUBOs_[i],sceneUBOMem_[i]);
            vkMapMemory(device_,sceneUBOMem_[i],0,sizeof(ProbeSceneUBO),0,&sceneMapped_[i]);
        }
    }

    void createDescriptorLayout()
    {
        std::array<VkDescriptorSetLayoutBinding,2> bs = {{
            {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,
             VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
            {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
        }};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount=2; dci.pBindings=bs.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_,&dci,nullptr,&sceneDSL_));
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize,2> sz = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAX_FRAMES},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES},
        }};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount=2; ci.pPoolSizes=sz.data(); ci.maxSets=MAX_FRAMES;
        VK_CHECK(vkCreateDescriptorPool(device_,&ci,nullptr,&pool_));
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> ls(MAX_FRAMES,sceneDSL_);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool=pool_; ai.descriptorSetCount=MAX_FRAMES; ai.pSetLayouts=ls.data();
        sceneSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_,&ai,sceneSets_.data()));
        for(int i=0;i<MAX_FRAMES;++i){
            VkDescriptorBufferInfo bi{sceneUBOs_[i],0,sizeof(ProbeSceneUBO)};
            VkDescriptorImageInfo ii{sampler_,cubeView_,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::array<VkWriteDescriptorSet,2> w = {{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,sceneSets_[i],0,0,1,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,nullptr,&bi},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,sceneSets_[i],1,0,1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,&ii,nullptr},
            }};
            vkUpdateDescriptorSets(device_,2,w.data(),0,nullptr);
        }
    }

    void createPipelines()
    {
        // Capture pipeline（probe_capture.vert/frag，push constant：viewProj）
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(CapturePC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.pushConstantRangeCount=1; lci.pPushConstantRanges=&pcr;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&captureLayout_));

            auto vert=createShaderModuleFromFile(device_,"probe_capture.vert.spv");
            auto frag=createShaderModuleFromFile(device_,"probe_capture.frag.spv");
            VkVertexInputBindingDescription bind{0,sizeof(SceneVertex),VK_VERTEX_INPUT_RATE_VERTEX};
            std::array<VkVertexInputAttributeDescription,3> attrs={{
                {0,0,VK_FORMAT_R32G32B32_SFLOAT,0},{1,0,VK_FORMAT_R32G32B32_SFLOAT,12},{2,0,VK_FORMAT_R32G32B32_SFLOAT,24}
            }};
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vi.vertexBindingDescriptionCount=1; vi.pVertexBindingDescriptions=&bind;
            vi.vertexAttributeDescriptionCount=3; vi.pVertexAttributeDescriptions=attrs.data();
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkViewport cvp{0,0,float(PROBE_SIZE),float(PROBE_SIZE),0,1};
            VkRect2D csc{{0,0},{PROBE_SIZE,PROBE_SIZE}};
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount=1; vps.pViewports=&cvp; vps.scissorCount=1; vps.pScissors=&csc;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_NONE; rs.lineWidth=1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable=VK_TRUE; ds.depthWriteEnable=VK_TRUE; ds.depthCompareOp=VK_COMPARE_OP_LESS;
            VkPipelineColorBlendAttachmentState blend{}; blend.colorWriteMask=0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount=1; cbs.pAttachments=&blend;
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vert; stages[0].pName="main";
            stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=frag; stages[1].pName="main";
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount=2; gci.pStages=stages;
            gci.pVertexInputState=&vi; gci.pInputAssemblyState=&ia;
            gci.pViewportState=&vps; gci.pRasterizationState=&rs;
            gci.pMultisampleState=&ms; gci.pDepthStencilState=&ds; gci.pColorBlendState=&cbs;
            gci.layout=captureLayout_; gci.renderPass=captureRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&capturePipeline_));
            vkDestroyShaderModule(device_,vert,nullptr); vkDestroyShaderModule(device_,frag,nullptr);
        }

        // Scene pipeline（probe_scene.vert/frag）
        {
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount=1; lci.pSetLayouts=&sceneDSL_;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&sceneLayout_));

            auto vert=createShaderModuleFromFile(device_,"probe_scene.vert.spv");
            auto frag=createShaderModuleFromFile(device_,"probe_scene.frag.spv");
            VkVertexInputBindingDescription bind{0,sizeof(SceneVertex),VK_VERTEX_INPUT_RATE_VERTEX};
            std::array<VkVertexInputAttributeDescription,3> attrs={{
                {0,0,VK_FORMAT_R32G32B32_SFLOAT,0},{1,0,VK_FORMAT_R32G32B32_SFLOAT,12},{2,0,VK_FORMAT_R32G32B32_SFLOAT,24}
            }};
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vi.vertexBindingDescriptionCount=1; vi.pVertexBindingDescriptions=&bind;
            vi.vertexAttributeDescriptionCount=3; vi.pVertexAttributeDescriptions=attrs.data();
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount=1; vps.scissorCount=1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_BACK_BIT;
            rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth=1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable=VK_TRUE; ds.depthWriteEnable=VK_TRUE; ds.depthCompareOp=VK_COMPARE_OP_LESS;
            VkPipelineColorBlendAttachmentState blend{}; blend.colorWriteMask=0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount=1; cbs.pAttachments=&blend;
            std::array<VkDynamicState,2> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dyns.dynamicStateCount=2; dyns.pDynamicStates=dyn.data();
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vert; stages[0].pName="main";
            stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=frag; stages[1].pName="main";
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount=2; gci.pStages=stages;
            gci.pVertexInputState=&vi; gci.pInputAssemblyState=&ia; gci.pViewportState=&vps;
            gci.pRasterizationState=&rs; gci.pMultisampleState=&ms; gci.pDepthStencilState=&ds;
            gci.pColorBlendState=&cbs; gci.pDynamicState=&dyns;
            gci.layout=sceneLayout_; gci.renderPass=sceneRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&scenePipeline_));
            vkDestroyShaderModule(device_,vert,nullptr); vkDestroyShaderModule(device_,frag,nullptr);
        }
    }

    void createCommandBuffers()
    {
        cmds_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool=cmdPool_; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=MAX_FRAMES;
        VK_CHECK(vkAllocateCommandBuffers(device_,&ai,cmds_.data()));
    }

    void createSyncObjects()
    {
        imgAvail_.resize(MAX_FRAMES); renderDone_.resize(MAX_FRAMES); inFlight_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fi.flags=VK_FENCE_CREATE_SIGNALED_BIT;
        for(int i=0;i<MAX_FRAMES;++i){
            VK_CHECK(vkCreateSemaphore(device_,&si,nullptr,&imgAvail_[i]));
            VK_CHECK(vkCreateSemaphore(device_,&si,nullptr,&renderDone_[i]));
            VK_CHECK(vkCreateFence(device_,&fi,nullptr,&inFlight_[i]));
        }
    }

    // ─── 探针捕获 ────────────────────────────────────────────────────────────

    void captureProbe()
    {
        vkDeviceWaitIdle(device_);
        VkCommandBuffer cmd=beginSingleTimeCommands(device_,cmdPool_);

        for(int f=0;f<6;++f){
            VkClearValue clears[2]{};
            clears[0].color.float32[0]=0.4f; clears[0].color.float32[1]=0.6f;
            clears[0].color.float32[2]=1.0f; clears[0].color.float32[3]=1.0f;
            clears[1].depthStencil.depth=1.0f;
            VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rbi.renderPass=captureRP_; rbi.framebuffer=captureFBs_[f];
            rbi.renderArea={0,0,PROBE_SIZE,PROBE_SIZE};
            rbi.clearValueCount=2; rbi.pClearValues=clears;
            vkCmdBeginRenderPass(cmd,&rbi,VK_SUBPASS_CONTENTS_INLINE);

            VkViewport vp{0,0,float(PROBE_SIZE),float(PROBE_SIZE),0,1};
            VkRect2D sc{{0,0},{PROBE_SIZE,PROBE_SIZE}};
            vkCmdSetViewport(cmd,0,1,&vp); vkCmdSetScissor(cmd,0,1,&sc);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,capturePipeline_);

            CapturePC pc{cubeFaceViewProj(f)};
            vkCmdPushConstants(cmd,captureLayout_,VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(CapturePC),&pc);

            VkDeviceSize z=0;
            vkCmdBindVertexBuffers(cmd,0,1,&envVB_,&z);
            vkCmdDraw(cmd,static_cast<uint32_t>(ENV_SCENE.size()),1,0,0);

            vkCmdEndRenderPass(cmd);
        }

        endSingleTimeCommands(device_,cmdPool_,gQueue_,cmd);
        std::cout << "✅ 探针捕获完成（6 个面，分辨率 " << PROBE_SIZE << "×" << PROBE_SIZE << "）\n";
    }

    // ─── 主循环 ──────────────────────────────────────────────────────────────

    void mainLoop()
    {
        while(!glfwWindowShouldClose(window_)){
            glfwPollEvents();
            interactive_.beginFrame(0.016f);
            buildUi();
            if(needRecapture_){
                captureProbe();
                needRecapture_=false;
            }
            drawFrame();
            interactive_.updateWindowTitle();
        }
        vkDeviceWaitIdle(device_);
    }

    void buildUi()
    {
        interactive_.buildDebugPanel("第59章：反射探针");
        ImGui::Separator();
        ImGui::Checkbox("显示 Cubemap 反射", &showReflect_);
        ImGui::SliderFloat("金属度 (metallic)",  &metallic_,  0.0f, 1.0f);
        ImGui::SliderFloat("粗糙度 (roughness)", &roughness_, 0.0f, 1.0f);
        if(ImGui::Button("重新捕获探针")) needRecapture_=true;
        ImGui::Separator();
        ImGui::TextWrapped(
            "Cubemap 分辨率：%d×%d，6 个面\n"
            "采样：reflect(viewDir, normal) 查 Cubemap\n"
            "Fresnel 控制反射强度（金属度↑ → 反射↑）",
            PROBE_SIZE, PROBE_SIZE);
    }

    void drawFrame()
    {
        vkWaitForFences(device_,1,&inFlight_[frame_],VK_TRUE,UINT64_MAX);
        uint32_t imgIdx=0;
        VkResult res=vkAcquireNextImageKHR(device_,swapchain_,UINT64_MAX,
                                           imgAvail_[frame_],VK_NULL_HANDLE,&imgIdx);
        if(res==VK_ERROR_OUT_OF_DATE_KHR){recreateSwapchain();return;}
        vkResetFences(device_,1,&inFlight_[frame_]);
        updateUBO(frame_);

        VkCommandBuffer cmd=cmds_[frame_];
        vkResetCommandBuffer(cmd,0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd,&bi));
        interactive_.beginGpuSection(cmd,frame_);

        VkClearValue clears[2]{};
        clears[0].color.float32[0]=0.15f; clears[0].color.float32[1]=0.15f;
        clears[0].color.float32[2]=0.2f;  clears[0].color.float32[3]=1.0f;
        clears[1].depthStencil.depth=1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass=sceneRP_; rbi.framebuffer=swapFBs_[imgIdx];
        rbi.renderArea.extent=extent_; rbi.clearValueCount=2; rbi.pClearValues=clears;
        vkCmdBeginRenderPass(cmd,&rbi,VK_SUBPASS_CONTENTS_INLINE);
        VkViewport vp{0,0,float(extent_.width),float(extent_.height),0,1};
        VkRect2D sc{{0,0},extent_};
        vkCmdSetViewport(cmd,0,1,&vp); vkCmdSetScissor(cmd,0,1,&sc);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,scenePipeline_);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,sceneLayout_,
                                0,1,&sceneSets_[frame_],0,nullptr);
        VkDeviceSize z=0;
        // 背景环境
        vkCmdBindVertexBuffers(cmd,0,1,&envVB_,&z);
        vkCmdDraw(cmd,static_cast<uint32_t>(ENV_SCENE.size()),1,0,0);
        // 中心金属球
        vkCmdBindVertexBuffers(cmd,0,1,&sphereVB_,&z);
        vkCmdDraw(cmd,sphereVerts_,1,0,0);

        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);

        interactive_.endGpuSection(cmd,frame_);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        VkPipelineStageFlags wait=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        si.waitSemaphoreCount=1; si.pWaitSemaphores=&imgAvail_[frame_]; si.pWaitDstStageMask=&wait;
        si.commandBufferCount=1; si.pCommandBuffers=&cmd;
        si.signalSemaphoreCount=1; si.pSignalSemaphores=&renderDone_[frame_];
        VK_CHECK(vkQueueSubmit(gQueue_,1,&si,inFlight_[frame_]));
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount=1; pi.pWaitSemaphores=&renderDone_[frame_];
        pi.swapchainCount=1; pi.pSwapchains=&swapchain_; pi.pImageIndices=&imgIdx;
        res=vkQueuePresentKHR(pQueue_,&pi);
        if(res==VK_ERROR_OUT_OF_DATE_KHR||res==VK_SUBOPTIMAL_KHR||resized_){
            resized_=false; recreateSwapchain();
        }
        interactive_.endFrame(frame_);
        frame_=(frame_+1)%MAX_FRAMES;
    }

    void updateUBO(uint32_t fi)
    {
        ProbeSceneUBO u{};
        u.model=glm::mat4(1.0f);
        u.view =interactive_.camera().viewMatrix();
        u.proj =glm::perspective(glm::radians(45.0f),
                                 float(extent_.width)/float(extent_.height),0.1f,50.0f);
        u.proj[1][1]*=-1;
        u.cameraPos=glm::vec4(interactive_.camera().eyePosition(),1);
        u.roughness=roughness_;
        u.metallic =showReflect_ ? metallic_ : 0.0f;
        std::memcpy(sceneMapped_[fi],&u,sizeof(u));
    }

    void recreateSwapchain()
    {
        int w=0,h=0; glfwGetFramebufferSize(window_,&w,&h);
        while(w==0||h==0){glfwGetFramebufferSize(window_,&w,&h);glfwWaitEvents();}
        vkDeviceWaitIdle(device_);
        for(auto fb:swapFBs_) vkDestroyFramebuffer(device_,fb,nullptr); swapFBs_.clear();
        destroyDepthResources(device_,screenDepth_);
        for(auto v:swapViews_) vkDestroyImageView(device_,v,nullptr);
        vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        createSwapchain(); createImageViews();
        screenDepth_=createDepthResources(physDev_,device_,extent_);
        createSwapFramebuffers();
        interactive_.onSwapchainRecreated(sceneRP_,swapFormat_,
                                          static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup()
    {
        vkDestroyDescriptorPool(device_,pool_,nullptr);
        vkDestroyDescriptorSetLayout(device_,sceneDSL_,nullptr);
        vkDestroyPipeline(device_,capturePipeline_,nullptr);
        vkDestroyPipelineLayout(device_,captureLayout_,nullptr);
        vkDestroyPipeline(device_,scenePipeline_,nullptr);
        vkDestroyPipelineLayout(device_,sceneLayout_,nullptr);
        vkDestroyRenderPass(device_,captureRP_,nullptr);
        vkDestroyRenderPass(device_,sceneRP_,nullptr);
        vkDestroySampler(device_,sampler_,nullptr);
        for(int f=0;f<6;++f){
            vkDestroyFramebuffer(device_,captureFBs_[f],nullptr);
            vkDestroyImageView(device_,cubeFaceViews_[f],nullptr);
        }
        vkDestroyImageView(device_,cubeView_,nullptr);
        vkDestroyImage(device_,cubeImage_,nullptr); vkFreeMemory(device_,cubeMem_,nullptr);
        vkDestroyImageView(device_,capDepthView_,nullptr);
        vkDestroyImage(device_,capDepthImage_,nullptr); vkFreeMemory(device_,capDepthMem_,nullptr);
        for(auto fb:swapFBs_) vkDestroyFramebuffer(device_,fb,nullptr);
        for(int i=0;i<MAX_FRAMES;++i){
            vkDestroyBuffer(device_,sceneUBOs_[i],nullptr); vkFreeMemory(device_,sceneUBOMem_[i],nullptr);
            vkDestroySemaphore(device_,imgAvail_[i],nullptr);
            vkDestroySemaphore(device_,renderDone_[i],nullptr);
            vkDestroyFence(device_,inFlight_[i],nullptr);
        }
        vkDestroyBuffer(device_,envVB_,nullptr);    vkFreeMemory(device_,envVBMem_,nullptr);
        vkDestroyBuffer(device_,sphereVB_,nullptr); vkFreeMemory(device_,sphereVBMem_,nullptr);
        destroyDepthResources(device_,screenDepth_);
        for(auto v:swapViews_) vkDestroyImageView(device_,v,nullptr);
        vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        vkDestroyCommandPool(device_,cmdPool_,nullptr);
        interactive_.shutdown(device_);
        vkDestroyDevice(device_,nullptr);
        vkDestroySurfaceKHR(instance_,surface_,nullptr);
        vkDestroyInstance(instance_,nullptr);
        glfwDestroyWindow(window_); glfwTerminate();
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════\n";
    std::cout << " 第59章：反射探针（Cubemap 环境反射）\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    std::cout << "Cubemap 分辨率：" << PROBE_SIZE << "×" << PROBE_SIZE << "，6 个面\n";
    std::cout << "中心金属球反射周围环境（红/蓝/绿柱 + 蓝天）\n";
    std::cout << "控制：LMB 旋转 | 滚轮缩放 | ImGui 调节金属度 | ESC 退出\n\n";
    try { Ch59App app; app.run(); std::cout<<"✅ 正常退出\n"; }
    catch(const std::exception& e){ std::cerr<<"❌ "<<e.what()<<"\n"; return 1; }
}
