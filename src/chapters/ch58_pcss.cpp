/**
 * @file ch58_pcss.cpp
 * @brief 第58章：软阴影 PCSS（Percentage-Closer Soft Shadows）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【硬阴影 vs 软阴影】
 *
 *  硬阴影（PCF）：所有遮挡都产生同样锐利的边缘，不物理真实。
 *
 *  PCSS 软阴影：光源越大、遮挡物离接收面越远 → 半影越宽。
 *
 *  PCSS 三步骤：
 *  1. Blocker Search：在当前像素周围找到所有遮挡者，计算平均遮挡深度 d_blocker
 *  2. Penumbra 估算：w_penumbra = (d_receiver - d_blocker) / d_blocker × lightSize
 *  3. PCF 滤波：用 w_penumbra 决定 PCF 采样半径（16-tap Poisson 分布）
 *
 * 【管线 · 2 个 Pass】
 *  Pass 1: Shadow Pass（pcss_shadow.vert，depth only，2048×2048）
 *  Pass 2: Scene Pass（pcss_scene.vert/frag，在 frag 执行 PCSS）
 *
 * 【ImGui 控制】
 *  Light Size（光源大小）、Shadow Bias、硬阴影↔PCSS 对比切换
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
constexpr uint32_t SHADOW_DIM = 2048;

struct SceneVertex { glm::vec3 pos; glm::vec3 normal; glm::vec3 color; };

struct SceneUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 lightMVP;
    alignas(16) glm::vec4 lightDir;
    alignas(16) glm::vec4 lightColor;
    alignas(16) glm::vec4 cameraPos;
    float lightSize;
    float shadowBias;
    float pad[2];
};

struct ShadowPC { glm::mat4 lightMVP; };

static const std::vector<SceneVertex> SCENE = {
    // 地面（阴影接收面）
    {{-5,-0.5f,-5},{0,1,0},{0.5f,0.48f,0.4f}}, {{ 5,-0.5f,-5},{0,1,0},{0.5f,0.48f,0.4f}},
    {{ 5,-0.5f, 5},{0,1,0},{0.5f,0.48f,0.4f}}, {{-5,-0.5f,-5},{0,1,0},{0.5f,0.48f,0.4f}},
    {{ 5,-0.5f, 5},{0,1,0},{0.5f,0.48f,0.4f}}, {{-5,-0.5f, 5},{0,1,0},{0.5f,0.48f,0.4f}},
    // 立柱（阴影投射者）
    {{-0.25f,-0.5f,-0.25f},{0,0,-1},{0.7f,0.65f,0.6f}}, {{ 0.25f,-0.5f,-0.25f},{0,0,-1},{0.7f,0.65f,0.6f}},
    {{ 0.25f, 3.0f,-0.25f},{0,0,-1},{0.7f,0.65f,0.6f}}, {{-0.25f,-0.5f,-0.25f},{0,0,-1},{0.7f,0.65f,0.6f}},
    {{ 0.25f, 3.0f,-0.25f},{0,0,-1},{0.7f,0.65f,0.6f}}, {{-0.25f, 3.0f,-0.25f},{0,0,-1},{0.7f,0.65f,0.6f}},
    // 低矮方块（离地面近，阴影边缘锐利）
    {{-3,-0.5f,-1},{0,0,-1},{0.4f,0.6f,0.8f}}, {{-2,-0.5f,-1},{0,0,-1},{0.4f,0.6f,0.8f}},
    {{-2, 0.5f,-1},{0,0,-1},{0.4f,0.6f,0.8f}}, {{-3,-0.5f,-1},{0,0,-1},{0.4f,0.6f,0.8f}},
    {{-2, 0.5f,-1},{0,0,-1},{0.4f,0.6f,0.8f}}, {{-3, 0.5f,-1},{0,0,-1},{0.4f,0.6f,0.8f}},
    // 高台（离地面远，阴影边缘柔和）
    {{ 2,-0.5f,-1},{0,0,-1},{0.8f,0.5f,0.3f}}, {{ 3,-0.5f,-1},{0,0,-1},{0.8f,0.5f,0.3f}},
    {{ 3, 2.5f,-1},{0,0,-1},{0.8f,0.5f,0.3f}}, {{ 2,-0.5f,-1},{0,0,-1},{0.8f,0.5f,0.3f}},
    {{ 3, 2.5f,-1},{0,0,-1},{0.8f,0.5f,0.3f}}, {{ 2, 2.5f,-1},{0,0,-1},{0.8f,0.5f,0.3f}},
};

class Ch58App {
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
    VkCommandPool  cmdPool_ = VK_NULL_HANDLE;

    // Shadow Map
    VkImage        shadowImage_  = VK_NULL_HANDLE;
    VkDeviceMemory shadowMem_    = VK_NULL_HANDLE;
    VkImageView    shadowView_   = VK_NULL_HANDLE;
    VkFramebuffer  shadowFB_     = VK_NULL_HANDLE;
    VkRenderPass   shadowRP_     = VK_NULL_HANDLE;

    // Scene
    VkRenderPass   sceneRP_ = VK_NULL_HANDLE;

    // Pipelines
    VkPipeline       shadowPipeline_  = VK_NULL_HANDLE;
    VkPipelineLayout shadowLayout_    = VK_NULL_HANDLE;
    VkPipeline       scenePipeline_   = VK_NULL_HANDLE;
    VkPipelineLayout sceneLayout_     = VK_NULL_HANDLE;
    VkDescriptorSetLayout sceneDSL_   = VK_NULL_HANDLE;

    VkDescriptorPool             pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneSets_;
    VkSampler shadowSampler_ = VK_NULL_HANDLE;

    VkBuffer       vb_    = VK_NULL_HANDLE; VkDeviceMemory vbMem_ = VK_NULL_HANDLE;
    std::vector<VkBuffer>       ubos_;
    std::vector<VkDeviceMemory> uboMem_;
    std::vector<void*>          uboMapped_;

    std::vector<VkCommandBuffer> cmds_;
    std::vector<VkSemaphore>     imgAvail_, renderDone_;
    std::vector<VkFence>         inFlight_;
    uint32_t frame_ = 0;
    bool     resized_ = false;

    InteractiveChapterTools interactive_;

    float lightSize_   = 5.0f;
    float shadowBias_  = 0.003f;
    glm::vec3 lightDir_{-1,-2,-1};
    bool  showHard_    = false;   // true = 硬阴影 (PCF radius=0)

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第58章：PCSS 软阴影", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch58App*>(glfwGetWindowUserPointer(w))->resized_ = true;
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
        createCommandPool();
        createShadowSampler(); createShadowMap();
        createRenderPasses(); createFramebuffers();
        uploadGeometry(); createUniformBuffers();
        createDescriptorLayout(); createDescriptorPool(); createDescriptorSets();
        createPipelines(); createCommandBuffers(); createSyncObjects();

        InteractiveInitInfo ii{};
        ii.window = window_; ii.instance = instance_;
        ii.physicalDevice = physDev_; ii.device = device_;
        ii.graphicsQueue = gQueue_; ii.queueFamily = queueIdx_.graphicsFamily.value();
        ii.renderPass = sceneRP_; ii.swapchainFormat = swapFormat_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setDistance(10.0f);
        interactive_.camera().setAngles(30.0f, 30.0f);
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

    void createShadowSampler()
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter=VK_FILTER_LINEAR; si.minFilter=VK_FILTER_LINEAR;
        si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.borderColor=VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;  // 边界外 = 无阴影
        si.maxLod=1.0f;
        VK_CHECK(vkCreateSampler(device_,&si,nullptr,&shadowSampler_));
    }

    void createShadowMap()
    {
        VkFormat fmt = findDepthFormat(physDev_);
        createImage(physDev_,device_,SHADOW_DIM,SHADOW_DIM,fmt,VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,shadowImage_,shadowMem_);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image=shadowImage_; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=fmt;
        vi.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};
        VK_CHECK(vkCreateImageView(device_,&vi,nullptr,&shadowView_));

        // 初始 layout → SHADER_READ_ONLY
        VkCommandBuffer cmd=beginSingleTimeCommands(device_,cmdPool_);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        b.image=shadowImage_; b.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};
        b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,0,nullptr,1,&b);
        endSingleTimeCommands(device_,cmdPool_,gQueue_,cmd);
    }

    void createRenderPasses()
    {
        // Shadow RP
        {
            VkAttachmentDescription a{};
            a.format=findDepthFormat(physDev_); a.samples=VK_SAMPLE_COUNT_1_BIT;
            a.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; a.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            a.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            a.finalLayout  =VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkAttachmentReference dr{0,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{}; sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.pDepthStencilAttachment=&dr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount=1; rpci.pAttachments=&a; rpci.subpassCount=1; rpci.pSubpasses=&sub;
            VK_CHECK(vkCreateRenderPass(device_,&rpci,nullptr,&shadowRP_));
        }
        // Scene RP
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
            VkSubpassDescription sub{}; sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount=1; sub.pColorAttachments=&cr; sub.pDepthStencilAttachment=&dr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount=2; rpci.pAttachments=a.data(); rpci.subpassCount=1; rpci.pSubpasses=&sub;
            VK_CHECK(vkCreateRenderPass(device_,&rpci,nullptr,&sceneRP_));
        }
    }

    void createFramebuffers()
    {
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass=shadowRP_; fci.attachmentCount=1; fci.pAttachments=&shadowView_;
        fci.width=SHADOW_DIM; fci.height=SHADOW_DIM; fci.layers=1;
        VK_CHECK(vkCreateFramebuffer(device_,&fci,nullptr,&shadowFB_));

        swapFBs_.resize(swapImages_.size());
        fci.renderPass=sceneRP_; fci.attachmentCount=2;
        fci.width=extent_.width; fci.height=extent_.height;
        for(size_t i=0;i<swapImages_.size();++i){
            std::array<VkImageView,2> atts={swapViews_[i],screenDepth_.view};
            fci.pAttachments=atts.data();
            VK_CHECK(vkCreateFramebuffer(device_,&fci,nullptr,&swapFBs_[i]));
        }
    }

    void uploadGeometry()
    {
        VkDeviceSize sz=sizeof(SceneVertex)*SCENE.size();
        VkBuffer st; VkDeviceMemory stm;
        createBuffer(physDev_,device_,sz,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,st,stm);
        void* d; vkMapMemory(device_,stm,0,sz,0,&d);
        std::memcpy(d,SCENE.data(),sz); vkUnmapMemory(device_,stm);
        createBuffer(physDev_,device_,sz,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,vb_,vbMem_);
        copyBuffer(device_,cmdPool_,gQueue_,st,vb_,sz);
        vkDestroyBuffer(device_,st,nullptr); vkFreeMemory(device_,stm,nullptr);
    }

    void createUniformBuffers()
    {
        ubos_.resize(MAX_FRAMES); uboMem_.resize(MAX_FRAMES); uboMapped_.resize(MAX_FRAMES);
        for(int i=0;i<MAX_FRAMES;++i){
            createBuffer(physDev_,device_,sizeof(SceneUBO),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         ubos_[i],uboMem_[i]);
            vkMapMemory(device_,uboMem_[i],0,sizeof(SceneUBO),0,&uboMapped_[i]);
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
            VkDescriptorBufferInfo bi{ubos_[i],0,sizeof(SceneUBO)};
            VkDescriptorImageInfo ii{shadowSampler_,shadowView_,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
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
        // Shadow pipeline（depth-only, no frag）
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(ShadowPC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.pushConstantRangeCount=1; lci.pPushConstantRanges=&pcr;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&shadowLayout_));

            auto vert=createShaderModuleFromFile(device_,"pcss_shadow.vert.spv");
            VkVertexInputBindingDescription bind{0,sizeof(SceneVertex),VK_VERTEX_INPUT_RATE_VERTEX};
            VkVertexInputAttributeDescription attr{0,0,VK_FORMAT_R32G32B32_SFLOAT,0};
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vi.vertexBindingDescriptionCount=1; vi.pVertexBindingDescriptions=&bind;
            vi.vertexAttributeDescriptionCount=1; vi.pVertexAttributeDescriptions=&attr;
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount=1; vps.scissorCount=1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_BACK_BIT;
            rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth=1.0f;
            rs.depthBiasEnable=VK_TRUE; rs.depthBiasConstantFactor=2.0f; rs.depthBiasSlopeFactor=1.5f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable=VK_TRUE; ds.depthWriteEnable=VK_TRUE; ds.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL;
            VkPipelineColorBlendStateCreateInfo noCbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            noCbs.attachmentCount=0;
            std::array<VkDynamicState,2> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dyns.dynamicStateCount=2; dyns.pDynamicStates=dyn.data();
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vert,"main"};
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount=1; gci.pStages=&stage;
            gci.pVertexInputState=&vi; gci.pInputAssemblyState=&ia; gci.pViewportState=&vps;
            gci.pRasterizationState=&rs; gci.pMultisampleState=&ms; gci.pDepthStencilState=&ds;
            gci.pColorBlendState=&noCbs; gci.pDynamicState=&dyns;
            gci.layout=shadowLayout_; gci.renderPass=shadowRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&shadowPipeline_));
            vkDestroyShaderModule(device_,vert,nullptr);
        }

        // Scene pipeline（PCSS）
        {
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount=1; lci.pSetLayouts=&sceneDSL_;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&sceneLayout_));

            auto vert=createShaderModuleFromFile(device_,"pcss_scene.vert.spv");
            auto frag=createShaderModuleFromFile(device_,"pcss_scene.frag.spv");
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
            rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_NONE; rs.lineWidth=1.0f;
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

    // ─── 主循环 ──────────────────────────────────────────────────────────────

    void mainLoop()
    {
        while(!glfwWindowShouldClose(window_)){
            glfwPollEvents();
            interactive_.beginFrame(0.016f);
            buildUi(); drawFrame();
            interactive_.updateWindowTitle();
        }
        vkDeviceWaitIdle(device_);
    }

    void buildUi()
    {
        interactive_.buildDebugPanel("第58章：PCSS 软阴影");
        ImGui::Separator();
        ImGui::Checkbox("硬阴影模式（光源大小=0）", &showHard_);
        if(!showHard_){
            ImGui::SliderFloat("光源大小（lightSize）", &lightSize_,  0.5f, 20.0f);
            ImGui::TextWrapped("光源越大、遮挡物离地面越远 → 半影越宽");
        }
        ImGui::SliderFloat("阴影偏移（bias）", &shadowBias_, 0.0f, 0.01f, "%.4f");
        ImGui::Separator();
        ImGui::TextWrapped(
            "PCSS 三步骤：\n"
            "1. Blocker Search：找遮挡者平均深度\n"
            "2. Penumbra 估算：半影半径 = lightSize × (d_recv-d_block)/d_block\n"
            "3. PCF 滤波：16-tap Poisson 圆盘");
    }

    glm::mat4 lightMVP() const
    {
        glm::mat4 lv = glm::lookAt(-glm::normalize(lightDir_)*18.0f, glm::vec3(0), glm::vec3(0,1,0));
        glm::mat4 lp = glm::ortho(-8.0f,8.0f,-8.0f,8.0f,1.0f,40.0f);
        lp[1][1]*=-1;
        return lp * lv;
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

        // ── Shadow Pass ──
        VkClearValue shadowClear{}; shadowClear.depthStencil.depth=1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass=shadowRP_; rbi.framebuffer=shadowFB_;
        rbi.renderArea={0,0,SHADOW_DIM,SHADOW_DIM};
        rbi.clearValueCount=1; rbi.pClearValues=&shadowClear;
        vkCmdBeginRenderPass(cmd,&rbi,VK_SUBPASS_CONTENTS_INLINE);
        VkViewport svp{0,0,float(SHADOW_DIM),float(SHADOW_DIM),0,1};
        VkRect2D ssc{{0,0},{SHADOW_DIM,SHADOW_DIM}};
        vkCmdSetViewport(cmd,0,1,&svp); vkCmdSetScissor(cmd,0,1,&ssc);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,shadowPipeline_);
        ShadowPC spc{lightMVP()};
        vkCmdPushConstants(cmd,shadowLayout_,VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(ShadowPC),&spc);
        VkDeviceSize z=0; vkCmdBindVertexBuffers(cmd,0,1,&vb_,&z);
        vkCmdDraw(cmd,static_cast<uint32_t>(SCENE.size()),1,0,0);
        vkCmdEndRenderPass(cmd);

        // ── Scene Pass ──
        VkClearValue clears[2]{};
        clears[0].color.float32[0]=0.6f; clears[0].color.float32[1]=0.65f;
        clears[0].color.float32[2]=0.75f; clears[0].color.float32[3]=1.0f;
        clears[1].depthStencil.depth=1.0f;
        rbi.renderPass=sceneRP_; rbi.framebuffer=swapFBs_[imgIdx];
        rbi.renderArea.extent=extent_; rbi.clearValueCount=2; rbi.pClearValues=clears;
        vkCmdBeginRenderPass(cmd,&rbi,VK_SUBPASS_CONTENTS_INLINE);
        VkViewport vp{0,0,float(extent_.width),float(extent_.height),0,1};
        VkRect2D sc{{0,0},extent_};
        vkCmdSetViewport(cmd,0,1,&vp); vkCmdSetScissor(cmd,0,1,&sc);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,scenePipeline_);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,sceneLayout_,
                                0,1,&sceneSets_[frame_],0,nullptr);
        vkCmdBindVertexBuffers(cmd,0,1,&vb_,&z);
        vkCmdDraw(cmd,static_cast<uint32_t>(SCENE.size()),1,0,0);
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
        glm::mat4 view=interactive_.camera().viewMatrix();
        glm::mat4 proj=glm::perspective(glm::radians(45.0f),
                                        float(extent_.width)/float(extent_.height),0.1f,100.0f);
        proj[1][1]*=-1;
        SceneUBO u{};
        u.model=glm::mat4(1.0f); u.view=view; u.proj=proj;
        u.lightMVP=lightMVP();
        u.lightDir=glm::vec4(glm::normalize(lightDir_),0);
        u.lightColor=glm::vec4(1.0f);
        u.cameraPos=glm::vec4(interactive_.camera().eyePosition(),1);
        u.lightSize=showHard_ ? 0.0f : lightSize_;
        u.shadowBias=shadowBias_;
        std::memcpy(uboMapped_[fi],&u,sizeof(u));
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
        createFramebuffers();
        interactive_.onSwapchainRecreated(sceneRP_,swapFormat_,
                                          static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup()
    {
        vkDestroyDescriptorPool(device_,pool_,nullptr);
        vkDestroyDescriptorSetLayout(device_,sceneDSL_,nullptr);
        vkDestroyPipeline(device_,shadowPipeline_,nullptr);
        vkDestroyPipelineLayout(device_,shadowLayout_,nullptr);
        vkDestroyPipeline(device_,scenePipeline_,nullptr);
        vkDestroyPipelineLayout(device_,sceneLayout_,nullptr);
        vkDestroyRenderPass(device_,shadowRP_,nullptr);
        vkDestroyRenderPass(device_,sceneRP_,nullptr);
        vkDestroySampler(device_,shadowSampler_,nullptr);
        vkDestroyImageView(device_,shadowView_,nullptr);
        vkDestroyImage(device_,shadowImage_,nullptr); vkFreeMemory(device_,shadowMem_,nullptr);
        vkDestroyFramebuffer(device_,shadowFB_,nullptr);
        for(auto fb:swapFBs_) vkDestroyFramebuffer(device_,fb,nullptr);
        for(int i=0;i<MAX_FRAMES;++i){
            vkDestroyBuffer(device_,ubos_[i],nullptr); vkFreeMemory(device_,uboMem_[i],nullptr);
            vkDestroySemaphore(device_,imgAvail_[i],nullptr);
            vkDestroySemaphore(device_,renderDone_[i],nullptr);
            vkDestroyFence(device_,inFlight_[i],nullptr);
        }
        vkDestroyBuffer(device_,vb_,nullptr); vkFreeMemory(device_,vbMem_,nullptr);
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
    std::cout << " 第58章：软阴影 PCSS（Percentage-Closer Soft Shadows）\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    std::cout << "对比：低物体阴影边缘锐利，高物体阴影边缘柔和\n";
    std::cout << "控制：LMB 旋转 | 滚轮缩放 | ImGui 调节光源大小 | ESC 退出\n\n";
    try { Ch58App app; app.run(); std::cout<<"✅ 正常退出\n"; }
    catch(const std::exception& e){ std::cerr<<"❌ "<<e.what()<<"\n"; return 1; }
}
