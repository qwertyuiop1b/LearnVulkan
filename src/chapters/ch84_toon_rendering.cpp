/**
 * @file ch84_toon_rendering.cpp
 * @brief 第84章：卡通渲染（Toon / Cel Shading）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【卡通渲染的核心技术】
 *
 *  真实感渲染 vs 卡通渲染：
 *    真实感：连续渐变光照（Phong / PBR）
 *    卡通感：离散色阶 + 硬边轮廓线
 *
 * 【两个 Pass 管线】
 *
 *  Pass 1：轮廓线（Outline Pass）
 *    - 绘制背面，顶点沿法线方向外扩
 *    - 技术：Inverted Hull（反转法线）
 *    - 着色器：toon_outline.vert / toon_outline.frag
 *
 *  Pass 2：主场景（Toon Shading）
 *    - 量化漫反射：floor(NdotL * bands) / bands
 *    - 二值化高光：step(threshold, NdotH^32)
 *    - 边缘光（Rim Light）：1-dot(N,V)，蓝紫色调
 *    - 着色器：toon.vert / toon.frag
 *
 * 【ImGui 控制】
 *    - 色阶数（2色/3色/4色/平滑渐变）
 *    - 轮廓线宽度（屏幕空间/世界空间切换）
 *    - 高光大小、边缘光强度
 *    - 轮廓线颜色（不只是黑色！）
 *    - 光源方向（仰角/方位角）
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
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH      = 900;
constexpr uint32_t HEIGHT     = 700;
constexpr int      MAX_FRAMES = 2;

// ─── 数据结构 ────────────────────────────────────────────────────────────────

struct SceneVertex { glm::vec3 pos; glm::vec3 normal; glm::vec3 color; };

struct ToonUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 lightDir;
    alignas(16) glm::vec4 lightColor;
    alignas(16) glm::vec4 cameraPos;
    float specularSize;
    float rimPower;
    float pad[2];
};

struct ToonPC {
    int   shadingBands;
    float shadowThreshold;
    float shadowSmooth;
    int   enableRim;
    int   enableSpecular;
    int   enableOutlineColor;
    float outlineColorDark;
    float pad;
};

struct OutlinePC {
    float outlineWidth;
    float outlineWidthNDC;
    int   useNDCWidth;
    float pad1;
    glm::vec4 outlineColor;
};

// ─── 程序化场景几何 ───────────────────────────────────────────────────────────

// 球体（卡通渲染的经典展示体）
static std::vector<SceneVertex> buildSphere(glm::vec3 center, float r, glm::vec3 color,
                                             int stacks=16, int slices=24)
{
    std::vector<SceneVertex> v;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            auto vtx = [&](int ii, int jj) -> SceneVertex {
                float phi   = float(ii) / stacks * glm::pi<float>();
                float theta = float(jj) / slices * 2.0f * glm::pi<float>();
                glm::vec3 n{sinf(phi)*cosf(theta), cosf(phi), sinf(phi)*sinf(theta)};
                return {center + n*r, n, color};
            };
            v.push_back(vtx(i,   j));
            v.push_back(vtx(i+1, j));
            v.push_back(vtx(i+1, j+1));
            v.push_back(vtx(i,   j));
            v.push_back(vtx(i+1, j+1));
            v.push_back(vtx(i,   j+1));
        }
    }
    return v;
}

// 地面
static std::vector<SceneVertex> buildGround()
{
    return {
        {{-4,-1,-4},{0,1,0},{0.55f,0.55f,0.5f}},
        {{ 4,-1,-4},{0,1,0},{0.55f,0.55f,0.5f}},
        {{ 4,-1, 4},{0,1,0},{0.55f,0.55f,0.5f}},
        {{-4,-1,-4},{0,1,0},{0.55f,0.55f,0.5f}},
        {{ 4,-1, 4},{0,1,0},{0.55f,0.55f,0.5f}},
        {{-4,-1, 4},{0,1,0},{0.55f,0.55f,0.5f}},
    };
}

// ─── App ─────────────────────────────────────────────────────────────────────

class Ch84App {
public:
    void run() { initWindow(); initVulkan(); mainLoop(); cleanup(); }

private:
    GLFWwindow*      window_  = nullptr;
    VkInstance       instance_= VK_NULL_HANDLE;
    VkSurfaceKHR     surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physDev_ = VK_NULL_HANDLE;
    VkDevice         device_  = VK_NULL_HANDLE;
    VkQueue          gQueue_  = VK_NULL_HANDLE;
    VkQueue          pQueue_  = VK_NULL_HANDLE;
    QueueFamilyIndices qIdx_{};
    VkSwapchainKHR   swapchain_= VK_NULL_HANDLE;
    std::vector<VkImage>       swapImages_;
    std::vector<VkImageView>   swapViews_;
    std::vector<VkFramebuffer> swapFBs_;
    VkFormat   swapFmt_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    DepthResources depth_{};
    VkCommandPool  cmdPool_    = VK_NULL_HANDLE;
    VkRenderPass   renderPass_ = VK_NULL_HANDLE;

    // Pipelines（Outline + Toon）
    VkPipeline       outlinePipeline_  = VK_NULL_HANDLE;
    VkPipelineLayout outlineLayout_    = VK_NULL_HANDLE;
    VkPipeline       toonPipeline_     = VK_NULL_HANDLE;
    VkPipelineLayout toonLayout_       = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_         = VK_NULL_HANDLE;
    VkDescriptorPool pool_             = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets_;

    // Buffers
    VkBuffer       sphereVB_  = VK_NULL_HANDLE; VkDeviceMemory sphereVBMem_  = VK_NULL_HANDLE;
    VkBuffer       groundVB_  = VK_NULL_HANDLE; VkDeviceMemory groundVBMem_  = VK_NULL_HANDLE;
    uint32_t       sphereCount_= 0, groundCount_= 0;
    std::vector<VkBuffer>       ubos_;
    std::vector<VkDeviceMemory> uboMem_;
    std::vector<void*>          uboMapped_;

    std::vector<VkCommandBuffer> cmds_;
    std::vector<VkSemaphore>     imgAvail_, renderDone_;
    std::vector<VkFence>         inFlight_;
    uint32_t frame_  = 0;
    bool     resized_= false;

    InteractiveChapterTools interactive_;
    float totalTime_= 0.0f;
    float deltaTime_= 0.016f;

    // ── ImGui 参数 ──────────────────────────────────────────────────────────
    int   shadingBands_       = 3;        // 色阶：2/3/4/0=平滑
    float shadowThreshold_    = 0.3f;
    float shadowSmooth_       = 0.05f;
    bool  enableRim_          = true;
    float rimPower_           = 3.0f;
    bool  enableSpecular_     = true;
    float specularSize_       = 0.4f;
    float outlineWidthNDC_    = 0.003f;
    bool  useNDCWidth_        = true;
    glm::vec4 outlineColor_   {0.05f, 0.05f, 0.05f, 1.0f};
    glm::vec3 lightDir_       = glm::normalize(glm::vec3(1,2,1));
    float lightElev_          = 45.0f;   // 仰角（度）
    float lightAzim_          = 45.0f;   // 方位角（度）
    glm::vec3 lightColor_     {1.0f, 0.95f, 0.85f};

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow()
    {
        glfwInit(); glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第84章：卡通渲染", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int){
            reinterpret_cast<Ch84App*>(glfwGetWindowUserPointer(w))->resized_=true;
        });
        interactive_.attachInput(window_);
    }

    void initVulkan()
    {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physDev_);
        createLogicalDevice(physDev_, surface_, device_, gQueue_, pQueue_, qIdx_);
        createSwapchain(); createImageViews();
        depth_ = createDepthResources(physDev_, device_, extent_);
        createCommandPool(); createRenderPass(); createFramebuffers();
        buildGeometry(); createUniformBuffers();
        createDescriptorLayout(); createDescriptorPool(); createDescriptorSets();
        createPipelines(); createCommandBuffers(); createSyncObjects();

        InteractiveInitInfo ii{};
        ii.window=window_; ii.instance=instance_;
        ii.physicalDevice=physDev_; ii.device=device_;
        ii.graphicsQueue=gQueue_; ii.queueFamily=qIdx_.graphicsFamily.value();
        ii.renderPass=renderPass_; ii.swapchainFormat=swapFmt_;
        ii.imageCount=static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight=MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setDistance(7.0f);
        interactive_.camera().setAngles(30.0f, 20.0f);
    }

    void createSwapchain()
    {
        auto d=querySwapChainSupport(physDev_,surface_);
        auto f=chooseSwapSurfaceFormat(d.formats);
        auto m=chooseSwapPresentMode(d.presentModes);
        extent_=chooseSwapExtent(d.capabilities,window_); swapFmt_=f.format;
        uint32_t cnt=d.capabilities.minImageCount+1;
        if(d.capabilities.maxImageCount) cnt=std::min(cnt,d.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        sci.surface=surface_; sci.minImageCount=cnt; sci.imageFormat=f.format;
        sci.imageColorSpace=f.colorSpace; sci.imageExtent=extent_; sci.imageArrayLayers=1;
        sci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        uint32_t qf[2]={qIdx_.graphicsFamily.value(),qIdx_.presentFamily.value()};
        if(qf[0]!=qf[1]){sci.imageSharingMode=VK_SHARING_MODE_CONCURRENT;sci.queueFamilyIndexCount=2;sci.pQueueFamilyIndices=qf;}
        else sci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform=d.capabilities.currentTransform;
        sci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; sci.presentMode=m; sci.clipped=VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_,&sci,nullptr,&swapchain_));
        uint32_t n=0; vkGetSwapchainImagesKHR(device_,swapchain_,&n,nullptr);
        swapImages_.resize(n); vkGetSwapchainImagesKHR(device_,swapchain_,&n,swapImages_.data());
    }

    void createImageViews()
    {
        swapViews_.resize(swapImages_.size());
        for(size_t i=0;i<swapImages_.size();++i){
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image=swapImages_[i]; vi.viewType=VK_IMAGE_VIEW_TYPE_2D;
            vi.format=swapFmt_; vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            VK_CHECK(vkCreateImageView(device_,&vi,nullptr,&swapViews_[i]));
        }
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex=qIdx_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_,&ci,nullptr,&cmdPool_));
    }

    void createRenderPass()
    {
        std::array<VkAttachmentDescription,2> atts{};
        atts[0].format=swapFmt_; atts[0].samples=VK_SAMPLE_COUNT_1_BIT;
        atts[0].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; atts[0].storeOp=VK_ATTACHMENT_STORE_OP_STORE;
        atts[0].initialLayout=VK_IMAGE_LAYOUT_UNDEFINED; atts[0].finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        atts[1].format=depth_.format; atts[1].samples=VK_SAMPLE_COUNT_1_BIT;
        atts[1].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; atts[1].storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[1].stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; atts[1].stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[1].initialLayout=VK_IMAGE_LAYOUT_UNDEFINED; atts[1].finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference cr{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference dr{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{}; sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount=1; sub.pColorAttachments=&cr; sub.pDepthStencilAttachment=&dr;
        VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpci.attachmentCount=2; rpci.pAttachments=atts.data(); rpci.subpassCount=1; rpci.pSubpasses=&sub;
        VK_CHECK(vkCreateRenderPass(device_,&rpci,nullptr,&renderPass_));
    }

    void createFramebuffers()
    {
        swapFBs_.resize(swapImages_.size());
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass=renderPass_; fci.attachmentCount=2;
        fci.width=extent_.width; fci.height=extent_.height; fci.layers=1;
        for(size_t i=0;i<swapImages_.size();++i){
            std::array<VkImageView,2> a={swapViews_[i],depth_.view};
            fci.pAttachments=a.data();
            VK_CHECK(vkCreateFramebuffer(device_,&fci,nullptr,&swapFBs_[i]));
        }
    }

    void buildGeometry()
    {
        auto upload=[&](const std::vector<SceneVertex>& v, VkBuffer& buf, VkDeviceMemory& mem)->uint32_t{
            VkDeviceSize sz=sizeof(SceneVertex)*v.size();
            VkBuffer st; VkDeviceMemory stm;
            createBuffer(physDev_,device_,sz,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,st,stm);
            void* d; vkMapMemory(device_,stm,0,sz,0,&d); std::memcpy(d,v.data(),sz); vkUnmapMemory(device_,stm);
            createBuffer(physDev_,device_,sz,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,buf,mem);
            copyBuffer(device_,cmdPool_,gQueue_,st,buf,sz);
            vkDestroyBuffer(device_,st,nullptr); vkFreeMemory(device_,stm,nullptr);
            return static_cast<uint32_t>(v.size());
        };

        // 3 个球：红色主角、蓝色、绿色
        std::vector<SceneVertex> spheres;
        auto s1 = buildSphere({0,0,0},   1.0f, {0.95f,0.3f,0.2f});   // 红
        auto s2 = buildSphere({-2.5f,0,0},0.7f, {0.2f,0.5f,0.9f});   // 蓝
        auto s3 = buildSphere({2.5f,0,0}, 0.7f, {0.2f,0.8f,0.3f});   // 绿
        spheres.insert(spheres.end(), s1.begin(), s1.end());
        spheres.insert(spheres.end(), s2.begin(), s2.end());
        spheres.insert(spheres.end(), s3.begin(), s3.end());
        sphereCount_ = upload(spheres, sphereVB_, sphereVBMem_);

        auto ground = buildGround();
        groundCount_ = upload(ground, groundVB_, groundVBMem_);
    }

    void createUniformBuffers()
    {
        ubos_.resize(MAX_FRAMES); uboMem_.resize(MAX_FRAMES); uboMapped_.resize(MAX_FRAMES);
        for(int i=0;i<MAX_FRAMES;++i){
            createBuffer(physDev_,device_,sizeof(ToonUBO),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         ubos_[i],uboMem_[i]);
            vkMapMemory(device_,uboMem_[i],0,sizeof(ToonUBO),0,&uboMapped_[i]);
        }
    }

    void createDescriptorLayout()
    {
        VkDescriptorSetLayoutBinding lb{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,
            VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount=1; dci.pBindings=&lb;
        VK_CHECK(vkCreateDescriptorSetLayout(device_,&dci,nullptr,&dsl_));
    }

    void createDescriptorPool()
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,MAX_FRAMES};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.poolSizeCount=1; pci.pPoolSizes=&ps; pci.maxSets=MAX_FRAMES;
        VK_CHECK(vkCreateDescriptorPool(device_,&pci,nullptr,&pool_));
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> ls(MAX_FRAMES,dsl_);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool=pool_; ai.descriptorSetCount=MAX_FRAMES; ai.pSetLayouts=ls.data();
        sets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_,&ai,sets_.data()));
        for(int i=0;i<MAX_FRAMES;++i){
            VkDescriptorBufferInfo bi{ubos_[i],0,sizeof(ToonUBO)};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,
                sets_[i],0,0,1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,nullptr,&bi};
            vkUpdateDescriptorSets(device_,1,&w,0,nullptr);
        }
    }

    void createPipelines()
    {
        VkVertexInputBindingDescription bind{0,sizeof(SceneVertex),VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription,3> attrs = {{
            {0,0,VK_FORMAT_R32G32B32_SFLOAT,0},
            {1,0,VK_FORMAT_R32G32B32_SFLOAT,12},
            {2,0,VK_FORMAT_R32G32B32_SFLOAT,24},
        }};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount=1; vi.pVertexBindingDescriptions=&bind;
        vi.vertexAttributeDescriptionCount=3; vi.pVertexAttributeDescriptions=attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount=1; vps.scissorCount=1;
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

        // ── Outline pipeline（背面剔除 → 前面，前面剔除 → 背面轮廓）──
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(OutlinePC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount=1; lci.pSetLayouts=&dsl_;
            lci.pushConstantRangeCount=1; lci.pPushConstantRanges=&pcr;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&outlineLayout_));

            auto vert=createShaderModuleFromFile(device_,"toon_outline.vert.spv");
            auto frag=createShaderModuleFromFile(device_,"toon_outline.frag.spv");

            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode=VK_POLYGON_MODE_FILL;
            rs.cullMode=VK_CULL_MODE_FRONT_BIT;   // ← 关键：剔除正面，只画背面
            rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth=1.0f;

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
            gci.layout=outlineLayout_; gci.renderPass=renderPass_;
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&outlinePipeline_));
            vkDestroyShaderModule(device_,vert,nullptr); vkDestroyShaderModule(device_,frag,nullptr);
        }

        // ── Toon pipeline（正常剔除背面，卡通着色）──
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(ToonPC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount=1; lci.pSetLayouts=&dsl_;
            lci.pushConstantRangeCount=1; lci.pPushConstantRanges=&pcr;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&toonLayout_));

            auto vert=createShaderModuleFromFile(device_,"toon.vert.spv");
            auto frag=createShaderModuleFromFile(device_,"toon.frag.spv");

            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode=VK_POLYGON_MODE_FILL;
            rs.cullMode=VK_CULL_MODE_BACK_BIT;
            rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth=1.0f;

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
            gci.layout=toonLayout_; gci.renderPass=renderPass_;
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&toonPipeline_));
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
        auto lastT = std::chrono::high_resolution_clock::now();
        while(!glfwWindowShouldClose(window_)){
            glfwPollEvents();
            auto now = std::chrono::high_resolution_clock::now();
            deltaTime_ = std::chrono::duration<float>(now-lastT).count(); lastT=now;
            totalTime_+= deltaTime_;
            interactive_.beginFrame(deltaTime_);
            buildUi(); drawFrame();
            interactive_.updateWindowTitle();
        }
        vkDeviceWaitIdle(device_);
    }

    void buildUi()
    {
        interactive_.buildDebugPanel("第84章：卡通渲染");
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1,0.85f,0.2f,1), "Cel Shading 参数");

        // 色阶选择
        const char* bandNames[] = {"平滑（Phong）","2 色阶","3 色阶","4 色阶"};
        int bandIdx = shadingBands_ == 0 ? 0 : shadingBands_-1;
        if (ImGui::Combo("色阶模式", &bandIdx, bandNames, 4))
            shadingBands_ = bandIdx == 0 ? 0 : bandIdx+1;

        ImGui::SliderFloat("阴影阈值",   &shadowThreshold_, 0.1f, 0.7f);
        ImGui::SliderFloat("边缘柔化",   &shadowSmooth_,    0.0f, 0.15f);
        ImGui::Separator();

        ImGui::Checkbox("启用高光",    &enableSpecular_);
        if(enableSpecular_)
            ImGui::SliderFloat("高光大小",  &specularSize_, 0.05f, 0.95f);

        ImGui::Checkbox("启用边缘光",  &enableRim_);
        if(enableRim_)
            ImGui::SliderFloat("边缘光强度",&rimPower_,     1.0f, 8.0f);

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.5f,0.8f,1,1), "轮廓线");
        ImGui::Checkbox("屏幕空间宽度（推荐）", &useNDCWidth_);
        ImGui::SliderFloat("轮廓线宽度", &outlineWidthNDC_, 0.001f, 0.01f, "%.4f");
        ImGui::ColorEdit4("轮廓线颜色", &outlineColor_.x);

        ImGui::Separator();
        ImGui::TextColored(ImVec4(1,0.6f,0.2f,1), "光源");
        bool lightChanged = false;
        lightChanged |= ImGui::SliderFloat("仰角（度）", &lightElev_, -90.0f, 90.0f);
        lightChanged |= ImGui::SliderFloat("方位角（度）",&lightAzim_,   0.0f, 360.0f);
        if(lightChanged) {
            float el=glm::radians(lightElev_), az=glm::radians(lightAzim_);
            lightDir_ = glm::normalize(glm::vec3(cosf(el)*sinf(az), sinf(el), cosf(el)*cosf(az)));
        }
        ImGui::ColorEdit3("光源颜色", &lightColor_.x);

        ImGui::Separator();
        ImGui::TextWrapped(
            "【技术说明】\n"
            "轮廓线：Pass1 绘制背面+法线外扩（Inverted Hull）\n"
            "卡通光照：Pass2 量化漫反射 floor(NdotL*bands)/bands\n"
            "高光：step(threshold, NdotH^32) 二值化\n"
            "边缘光：1-dot(N,V) → 蓝紫色调 rim");
    }

    void drawFrame()
    {
        vkWaitForFences(device_,1,&inFlight_[frame_],VK_TRUE,UINT64_MAX);
        uint32_t idx=0;
        VkResult r=vkAcquireNextImageKHR(device_,swapchain_,UINT64_MAX,imgAvail_[frame_],VK_NULL_HANDLE,&idx);
        if(r==VK_ERROR_OUT_OF_DATE_KHR){recreate();return;}
        vkResetFences(device_,1,&inFlight_[frame_]);
        updateUBO(frame_);

        VkCommandBuffer cmd=cmds_[frame_];
        vkResetCommandBuffer(cmd,0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd,&bi));
        interactive_.beginGpuSection(cmd,frame_);

        VkClearValue clears[2]{};
        clears[0].color.float32[0]=0.85f; clears[0].color.float32[1]=0.88f;
        clears[0].color.float32[2]=0.92f; clears[0].color.float32[3]=1.0f;
        clears[1].depthStencil.depth=1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass=renderPass_; rbi.framebuffer=swapFBs_[idx];
        rbi.renderArea.extent=extent_; rbi.clearValueCount=2; rbi.pClearValues=clears;
        vkCmdBeginRenderPass(cmd,&rbi,VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp{0,0,float(extent_.width),float(extent_.height),0,1};
        VkRect2D sc{{0,0},extent_};
        vkCmdSetViewport(cmd,0,1,&vp); vkCmdSetScissor(cmd,0,1,&sc);

        // ── Pass 1: 轮廓线（Outline，背面外扩）──────────────────────────────
        {
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,outlinePipeline_);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,outlineLayout_,
                                    0,1,&sets_[frame_],0,nullptr);
            OutlinePC opc{};
            opc.outlineWidth    = 0.03f;
            opc.outlineWidthNDC = outlineWidthNDC_;
            opc.useNDCWidth     = useNDCWidth_ ? 1 : 0;
            opc.outlineColor    = outlineColor_;
            vkCmdPushConstants(cmd,outlineLayout_,
                               VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,sizeof(OutlinePC),&opc);
            VkDeviceSize zero=0;
            vkCmdBindVertexBuffers(cmd,0,1,&sphereVB_,&zero);
            vkCmdDraw(cmd,sphereCount_,1,0,0);
            vkCmdBindVertexBuffers(cmd,0,1,&groundVB_,&zero);
            vkCmdDraw(cmd,groundCount_,1,0,0);
        }

        // ── Pass 2: 主场景（Toon Shading）────────────────────────────────────
        {
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,toonPipeline_);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,toonLayout_,
                                    0,1,&sets_[frame_],0,nullptr);
            ToonPC tpc{};
            tpc.shadingBands    = shadingBands_;
            tpc.shadowThreshold = shadowThreshold_;
            tpc.shadowSmooth    = shadowSmooth_;
            tpc.enableRim       = enableRim_ ? 1 : 0;
            tpc.enableSpecular  = enableSpecular_ ? 1 : 0;
            tpc.enableOutlineColor = 0;
            tpc.outlineColorDark   = 0.5f;
            vkCmdPushConstants(cmd,toonLayout_,
                               VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,sizeof(ToonPC),&tpc);
            VkDeviceSize zero=0;
            vkCmdBindVertexBuffers(cmd,0,1,&sphereVB_,&zero);
            vkCmdDraw(cmd,sphereCount_,1,0,0);
            vkCmdBindVertexBuffers(cmd,0,1,&groundVB_,&zero);
            vkCmdDraw(cmd,groundCount_,1,0,0);
        }

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
        pi.swapchainCount=1; pi.pSwapchains=&swapchain_; pi.pImageIndices=&idx;
        r=vkQueuePresentKHR(pQueue_,&pi);
        if(r==VK_ERROR_OUT_OF_DATE_KHR||r==VK_SUBOPTIMAL_KHR||resized_){resized_=false;recreate();}
        interactive_.endFrame(frame_);
        frame_=(frame_+1)%MAX_FRAMES;
    }

    void updateUBO(uint32_t fi)
    {
        ToonUBO u{};
        u.model     = glm::mat4(1.0f);
        u.view      = interactive_.camera().viewMatrix();
        u.proj      = glm::perspective(glm::radians(45.0f),
                                       float(extent_.width)/float(extent_.height), 0.1f, 100.0f);
        u.proj[1][1]*=-1;
        u.lightDir  = glm::vec4(glm::normalize(lightDir_), 0.0f);
        u.lightColor= glm::vec4(lightColor_, 1.0f);
        u.cameraPos = glm::vec4(interactive_.camera().eyePosition(), 1.0f);
        u.specularSize = specularSize_;
        u.rimPower     = rimPower_;
        std::memcpy(uboMapped_[fi],&u,sizeof(u));
    }

    void recreate()
    {
        int w=0,h=0; glfwGetFramebufferSize(window_,&w,&h);
        while(w==0||h==0){glfwGetFramebufferSize(window_,&w,&h);glfwWaitEvents();}
        vkDeviceWaitIdle(device_);
        for(auto fb:swapFBs_) vkDestroyFramebuffer(device_,fb,nullptr); swapFBs_.clear();
        destroyDepthResources(device_,depth_);
        for(auto v:swapViews_) vkDestroyImageView(device_,v,nullptr);
        vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        createSwapchain(); createImageViews();
        depth_=createDepthResources(physDev_,device_,extent_);
        createFramebuffers();
        interactive_.onSwapchainRecreated(renderPass_,swapFmt_,
                                          static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup()
    {
        vkDestroyDescriptorPool(device_,pool_,nullptr);
        vkDestroyDescriptorSetLayout(device_,dsl_,nullptr);
        vkDestroyPipeline(device_,outlinePipeline_,nullptr);
        vkDestroyPipelineLayout(device_,outlineLayout_,nullptr);
        vkDestroyPipeline(device_,toonPipeline_,nullptr);
        vkDestroyPipelineLayout(device_,toonLayout_,nullptr);
        vkDestroyRenderPass(device_,renderPass_,nullptr);
        for(auto fb:swapFBs_) vkDestroyFramebuffer(device_,fb,nullptr);
        destroyDepthResources(device_,depth_);
        for(int i=0;i<MAX_FRAMES;++i){
            vkDestroyBuffer(device_,ubos_[i],nullptr); vkFreeMemory(device_,uboMem_[i],nullptr);
            vkDestroySemaphore(device_,imgAvail_[i],nullptr);
            vkDestroySemaphore(device_,renderDone_[i],nullptr);
            vkDestroyFence(device_,inFlight_[i],nullptr);
        }
        vkDestroyBuffer(device_,sphereVB_,nullptr); vkFreeMemory(device_,sphereVBMem_,nullptr);
        vkDestroyBuffer(device_,groundVB_,nullptr); vkFreeMemory(device_,groundVBMem_,nullptr);
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
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第84章：卡通渲染（Toon / Cel Shading）\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    std::cout << "技术：量化漫反射 + 硬高光 + 边缘光 + 轮廓线（Inverted Hull）\n";
    std::cout << "控制：LMB 旋转 | RMB 平移 | 滚轮缩放 | ImGui 调节参数 | ESC 退出\n\n";
    try { Ch84App app; app.run(); std::cout<<"✅ 正常退出\n"; }
    catch(const std::exception& e){ std::cerr<<"❌ "<<e.what()<<"\n"; return 1; }
}
