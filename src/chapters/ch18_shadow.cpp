/**
 * @file ch18_shadow.cpp
 * @brief 第18章：Shadow Mapping（阴影映射）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【Shadow Mapping 原理】
 *
 *  两趟渲染（Two-Pass Rendering）：
 *
 *  Pass 1（阴影通道）：
 *    从光源视角渲染场景 → 只写深度值到 Shadow Map（深度纹理）
 *    光源遮住的地方：shadow map 深度小（近处）
 *    光源照到的地方：shadow map 深度大（远处）
 *
 *  Pass 2（场景通道）：
 *    从相机视角渲染，每个片段：
 *    1. 将世界坐标变换到光源空间
 *    2. 对比当前深度与 Shadow Map 中的深度
 *    3. 如果当前深度 > 存储深度 + bias → 在阴影中
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  Shadow Map：1024×1024 深度纹理                         │
 *  │                                                         │
 *  │  光源视角                                               │
 *  │  ┌──────────────────┐                                  │
 *  │  │ [深度值小] 柱子   │                                  │
 *  │  │ [深度值大] 地面   │                                  │
 *  │  └──────────────────┘                                  │
 *  │                                                         │
 *  │  相机视角 - 地面上采样 Shadow Map                        │
 *  │    currentDepth > shadowMapDepth? → 阴影 → 只有 ambient │
 * └─────────────────────────────────────────────────────────┘
 *
 * 【PCF（Percentage Closer Filtering）】
 *
 *  硬阴影：每像素采样1次Shadow Map，边界锯齿明显
 *  PCF：对3×3邻域采样9次取平均，产生柔和半影
 *
 * 【Shadow Acne（自遮挡伪影）】
 *
 *  由于Shadow Map精度有限，斜面会错误地遮挡自身。
 *  解决：添加深度偏移（bias），常用：
 *    bias = max(0.005*(1-dot(N,L)), 0.001);
 *
 * 【Vulkan 实现要点】
 *
 *  Shadow Map 纹理格式：VK_FORMAT_D32_SFLOAT（32位深度）
 *  Shadow Map 采样器：
 *    - addressMode = CLAMP_TO_BORDER（超出边界不在阴影中）
 *    - borderColor  = FLOAT_OPAQUE_WHITE（深度=1.0，无阴影）
 *    - compareEnable = VK_TRUE（PCF 硬件支持）
 *    - compareOp     = LESS_OR_EQUAL
 *  Shadow Pass RenderPass：只有深度附件，无颜色附件
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan_tutorial/utils.hpp>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>

constexpr uint32_t WIDTH       = 800;
constexpr uint32_t HEIGHT      = 600;
constexpr int      MAX_FRAMES  = 2;
constexpr uint32_t SHADOW_DIM  = 1024;   // Shadow Map 分辨率

struct Vertex { glm::vec3 pos; glm::vec3 normal; glm::vec3 color; };

// 场景：地面 + 3根立柱
static const std::vector<Vertex> VERTICES = {
    // 地面（大平面）
    {{-3,-0.5f,-3},{0,1,0},{0.5f,0.5f,0.5f}}, {{ 3,-0.5f,-3},{0,1,0},{0.5f,0.5f,0.5f}},
    {{ 3,-0.5f, 3},{0,1,0},{0.5f,0.5f,0.5f}}, {{-3,-0.5f,-3},{0,1,0},{0.5f,0.5f,0.5f}},
    {{ 3,-0.5f, 3},{0,1,0},{0.5f,0.5f,0.5f}}, {{-3,-0.5f, 3},{0,1,0},{0.5f,0.5f,0.5f}},
    // 立柱1（红）
    {{-0.2f,-0.5f,-0.2f},{0,0,1},{1,0.2f,0.2f}}, {{ 0.2f,-0.5f,-0.2f},{0,0,1},{1,0.2f,0.2f}},
    {{ 0.2f, 1.5f,-0.2f},{0,0,1},{1,0.2f,0.2f}}, {{-0.2f,-0.5f,-0.2f},{0,0,1},{1,0.2f,0.2f}},
    {{ 0.2f, 1.5f,-0.2f},{0,0,1},{1,0.2f,0.2f}}, {{-0.2f, 1.5f,-0.2f},{0,0,1},{1,0.2f,0.2f}},
    // 立柱2（绿，偏左）
    {{-1.5f,-0.5f,0.3f},{1,0,0},{0.2f,1,0.2f}}, {{-1.1f,-0.5f,0.3f},{1,0,0},{0.2f,1,0.2f}},
    {{-1.1f, 2.0f,0.3f},{1,0,0},{0.2f,1,0.2f}}, {{-1.5f,-0.5f,0.3f},{1,0,0},{0.2f,1,0.2f}},
    {{-1.1f, 2.0f,0.3f},{1,0,0},{0.2f,1,0.2f}}, {{-1.5f, 2.0f,0.3f},{1,0,0},{0.2f,1,0.2f}},
};

struct SceneUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::mat4 lightSpaceMatrix;
    alignas(16) glm::vec4 lightDir;
};

struct ShadowPushConstants {
    glm::mat4 lightSpaceMatrix;
};

class Ch18App {
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

    // ── Pass 1：Shadow Pass ─────────────────────────────────────────────────
    VkImage        shadowImage_       = VK_NULL_HANDLE;  ///< Shadow Map 深度纹理
    VkDeviceMemory shadowImageMemory_ = VK_NULL_HANDLE;
    VkImageView    shadowImageView_   = VK_NULL_HANDLE;
    VkSampler      shadowSampler_     = VK_NULL_HANDLE;  ///< 带 compareOp 的采样器
    VkFramebuffer  shadowFramebuffer_ = VK_NULL_HANDLE;
    VkRenderPass   shadowRenderPass_  = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline       shadowPipeline_       = VK_NULL_HANDLE;

    // ── Pass 2：Scene Pass ──────────────────────────────────────────────────
    VkRenderPass          sceneRenderPass_  = VK_NULL_HANDLE;
    VkDescriptorSetLayout sceneSetLayout_   = VK_NULL_HANDLE;
    VkPipelineLayout      scenePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline            scenePipeline_       = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool_   = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneDescSets_;

    VkImage        depthImage_       = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
    VkImageView    depthImageView_   = VK_NULL_HANDLE;
    VkFormat       depthFormat_      = VK_FORMAT_UNDEFINED;

    VkBuffer       vertexBuffer_     = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_     = VK_NULL_HANDLE;
    std::vector<VkBuffer>       sceneUBOs_;
    std::vector<VkDeviceMemory> sceneUBOMemories_;
    std::vector<void*>          sceneUBOMapped_;

    std::vector<VkImage>         swapchainImages_;
    std::vector<VkImageView>     swapchainImageViews_;
    std::vector<VkFramebuffer>   sceneFramebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkFormat                     swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D                   swapchainExtent_{};
    QueueFamilyIndices           queueIndices_;

    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence>     inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool     resized_      = false;

    // 光源方向（平行光）
    glm::vec3 lightDir_ = glm::normalize(glm::vec3(-1.0f, -2.0f, -1.0f));

    void initWindow()
    {
        glfwInit();glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);
        window_=glfwCreateWindow(WIDTH,HEIGHT,"Ch18 - Shadow Mapping（PCF 软阴影）",nullptr,nullptr);
        glfwSetWindowUserPointer(window_,this);
        glfwSetFramebufferSizeCallback(window_,[](GLFWwindow*w,int,int){reinterpret_cast<Ch18App*>(glfwGetWindowUserPointer(w))->resized_=true;});
    }

    void initVulkan()
    {
        createInstance();createSurface();pickPhysicalDevice();createLogicalDevice();
        createSwapchain();createImageViews();
        depthFormat_=findDepthFormat();
        createShadowResources();       // ← Step 1: Shadow Map 纹理和采样器
        createShadowRenderPass();      // ← Step 2: 只有深度附件的 RenderPass
        createShadowPipeline();        // ← Step 3: 只有顶点着色器的管线
        createSceneRenderPass();
        createSceneDescriptorSetLayout();
        createScenePipeline();
        createDepthResources();createSceneFramebuffers();
        createCommandPool();createVertexBuffer();
        createSceneUBOs();
        createDescriptorPool();createSceneDescriptorSets();
        createCommandBuffers();createSyncObjects();
        std::cout<<"\n✅ Shadow Mapping 初始化完成！\n";
        std::cout<<"📐 Shadow Map 分辨率："<<SHADOW_DIM<<"×"<<SHADOW_DIM<<"\n";
        std::cout<<"💡 光源方向："<<lightDir_.x<<", "<<lightDir_.y<<", "<<lightDir_.z<<"\n";
        std::cout<<"🔦 PCF 软阴影（3×3 采样，9次）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 1: 创建 Shadow Map（深度纹理 + 特殊采样器）
    // ═══════════════════════════════════════════════════════════════════════

    void createShadowResources()
    {
        // Shadow Map 是一张深度图像
        VkImageCreateInfo imgCI{};
        imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgCI.imageType     = VK_IMAGE_TYPE_2D;
        imgCI.extent        = {SHADOW_DIM, SHADOW_DIM, 1};
        imgCI.mipLevels     = 1; imgCI.arrayLayers = 1;
        imgCI.format        = VK_FORMAT_D32_SFLOAT;
        imgCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgCI.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT;  // 作为纹理供场景Pass采样
        imgCI.samples       = VK_SAMPLE_COUNT_1_BIT;
        VK_CHECK(vkCreateImage(device_,&imgCI,nullptr,&shadowImage_));

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device_,shadowImage_,&mr);
        VkMemoryAllocateInfo allocCI{};allocCI.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocCI.allocationSize=mr.size;allocCI.memoryTypeIndex=findMemoryType(mr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device_,&allocCI,nullptr,&shadowImageMemory_));
        VK_CHECK(vkBindImageMemory(device_,shadowImage_,shadowImageMemory_,0));

        // Shadow Map ImageView
        VkImageViewCreateInfo viewCI{};viewCI.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCI.image=shadowImage_;viewCI.viewType=VK_IMAGE_VIEW_TYPE_2D;
        viewCI.format=VK_FORMAT_D32_SFLOAT;
        viewCI.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};
        VK_CHECK(vkCreateImageView(device_,&viewCI,nullptr,&shadowImageView_));

        // ── Shadow Map 采样器（专为阴影比较设计）────────────────────────────
        VkSamplerCreateInfo sampCI{};sampCI.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampCI.magFilter=VK_FILTER_LINEAR;sampCI.minFilter=VK_FILTER_LINEAR;
        // CLAMP_TO_BORDER + WHITE：超出阴影贴图的区域深度=1（无阴影）
        sampCI.addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampCI.addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampCI.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampCI.borderColor=VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        // compareEnable：启用深度比较采样（硬件 PCF 支持）
        sampCI.compareEnable=VK_FALSE;  // 我们在 shader 中手动实现 PCF
        sampCI.compareOp=VK_COMPARE_OP_LESS_OR_EQUAL;
        sampCI.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR;
        VK_CHECK(vkCreateSampler(device_,&sampCI,nullptr,&shadowSampler_));

        std::cout<<"✅ Shadow Map 已创建（"<<SHADOW_DIM<<"×"<<SHADOW_DIM<<" D32_SFLOAT）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 2: Shadow Pass RenderPass（只有深度附件）
    // ═══════════════════════════════════════════════════════════════════════

    void createShadowRenderPass()
    {
        VkAttachmentDescription depthAtt{};
        depthAtt.format         = VK_FORMAT_D32_SFLOAT;
        depthAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
        depthAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;    // 每次清除
        depthAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;   // 保留深度值（供 Pass2 读取）
        depthAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAtt.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // 给场景Pass读取

        VkAttachmentReference depthRef{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 0;        // ← 无颜色附件！
        subpass.pDepthStencilAttachment = &depthRef;

        // 依赖：Shadow Pass 写完 → Scene Pass 读取
        std::array<VkSubpassDependency,2> deps{};
        deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
        deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[0].dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        deps[1].srcSubpass    = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo rpi{};rpi.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount=1;rpi.pAttachments=&depthAtt;
        rpi.subpassCount=1;rpi.pSubpasses=&subpass;
        rpi.dependencyCount=static_cast<uint32_t>(deps.size());rpi.pDependencies=deps.data();
        VK_CHECK(vkCreateRenderPass(device_,&rpi,nullptr,&shadowRenderPass_));

        // Shadow Framebuffer（绑定 shadow map depth view）
        VkFramebufferCreateInfo fbCI{};fbCI.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbCI.renderPass=shadowRenderPass_;fbCI.attachmentCount=1;
        fbCI.pAttachments=&shadowImageView_;
        fbCI.width=SHADOW_DIM;fbCI.height=SHADOW_DIM;fbCI.layers=1;
        VK_CHECK(vkCreateFramebuffer(device_,&fbCI,nullptr,&shadowFramebuffer_));

        std::cout<<"✅ Shadow RenderPass 创建（仅深度，无颜色）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 3: Shadow Pipeline（只有顶点着色器，片段着色器为空）
    // ═══════════════════════════════════════════════════════════════════════

    void createShadowPipeline()
    {
        VkShaderModule vert=createShaderModuleFromFile(device_,"shadow_depth.vert.spv");
        VkShaderModule frag=createShaderModuleFromFile(device_,"shadow_depth.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vert,"main",nullptr};
        stages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,frag,"main",nullptr};

        // 顶点输入：只需要位置
        VkVertexInputBindingDescription bind{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attr{0,0,VK_FORMAT_R32G32B32_SFLOAT,0};
        VkPipelineVertexInputStateCreateInfo vi{};vi.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount=1;vi.pVertexBindingDescriptions=&bind;
        vi.vertexAttributeDescriptionCount=1;vi.pVertexAttributeDescriptions=&attr;

        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,nullptr,0,VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,VK_FALSE};

        // Shadow Map 视口（不同于交换链大小）
        VkViewport shadowVP{0,0,(float)SHADOW_DIM,(float)SHADOW_DIM,0,1};
        VkRect2D   shadowSC{{0,0},{SHADOW_DIM,SHADOW_DIM}};
        VkPipelineViewportStateCreateInfo vs{};vs.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vs.viewportCount=1;vs.pViewports=&shadowVP;vs.scissorCount=1;vs.pScissors=&shadowSC;

        VkPipelineRasterizationStateCreateInfo rs{};rs.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode=VK_POLYGON_MODE_FILL;rs.lineWidth=1.0f;
        rs.cullMode=VK_CULL_MODE_BACK_BIT; // 渲染背面，避免 peter panning（物体与阴影分离）
        rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;
        // 深度偏移（减少 shadow acne）
        rs.depthBiasEnable=VK_TRUE;
        rs.depthBiasConstantFactor=1.25f;  // 恒定偏移
        rs.depthBiasSlopeFactor=1.75f;     // 斜率相关偏移（斜面需要更大偏移）

        VkPipelineMultisampleStateCreateInfo ms{};ms.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};ds.sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable=VK_TRUE;ds.depthWriteEnable=VK_TRUE;ds.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL;

        // Shadow Pass 无颜色附件 → colorBlend 为空
        VkPipelineColorBlendStateCreateInfo cb{};cb.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount=0;

        // Push Constant：光源空间矩阵
        VkPushConstantRange pcRange{};pcRange.stageFlags=VK_SHADER_STAGE_VERTEX_BIT;
        pcRange.offset=0;pcRange.size=sizeof(ShadowPushConstants);

        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_,&pli,nullptr,&shadowPipelineLayout_));

        VkGraphicsPipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount=2;pi.pStages=stages;
        pi.pVertexInputState=&vi;pi.pInputAssemblyState=&ia;pi.pViewportState=&vs;
        pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pDepthStencilState=&ds;
        pi.pColorBlendState=&cb;pi.layout=shadowPipelineLayout_;pi.renderPass=shadowRenderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&shadowPipeline_));
        vkDestroyShaderModule(device_,frag,nullptr);vkDestroyShaderModule(device_,vert,nullptr);
        std::cout<<"✅ Shadow 管线创建（深度偏移 slope=1.75）\n";
    }

    // 计算光源空间矩阵（正交投影，覆盖整个场景）
    glm::mat4 getLightSpaceMatrix()
    {
        glm::mat4 lightView = glm::lookAt(
            -lightDir_ * 5.0f,    // 光源位置
            glm::vec3(0,0,0),     // 看向场景中心
            glm::vec3(0,1,0));
        // 正交投影（平行光）
        glm::mat4 lightProj = glm::ortho(-5.0f,5.0f,-5.0f,5.0f,0.1f,15.0f);
        lightProj[1][1] *= -1;   // Vulkan Y 轴翻转
        return lightProj * lightView;
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo bi{};bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd,&bi));

        glm::mat4 lightSpaceMat = getLightSpaceMatrix();

        // ════════════════════════════════════════════════════════════════════
        // Pass 1：Shadow Pass（从光源视角渲染，生成深度图）
        // ════════════════════════════════════════════════════════════════════
        {
            VkClearValue shadowClear{};shadowClear.depthStencil={1.0f,0};
            VkRenderPassBeginInfo rp{};rp.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp.renderPass=shadowRenderPass_;rp.framebuffer=shadowFramebuffer_;
            rp.renderArea={{0,0},{SHADOW_DIM,SHADOW_DIM}};
            rp.clearValueCount=1;rp.pClearValues=&shadowClear;

            vkCmdBeginRenderPass(cmd,&rp,VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,shadowPipeline_);

            // Push Constant：传递光源空间矩阵
            ShadowPushConstants pc{lightSpaceMat};
            vkCmdPushConstants(cmd,shadowPipelineLayout_,
                VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(pc),&pc);

            VkBuffer vb[]={vertexBuffer_};VkDeviceSize off[]={0};
            vkCmdBindVertexBuffers(cmd,0,1,vb,off);
            vkCmdDraw(cmd,static_cast<uint32_t>(VERTICES.size()),1,0,0);

            vkCmdEndRenderPass(cmd);
        }

        // ════════════════════════════════════════════════════════════════════
        // Pass 2：Scene Pass（正常渲染 + 阴影比较）
        // ════════════════════════════════════════════════════════════════════
        {
            std::array<VkClearValue,2> clears{};
            clears[0].color.float32[0]=0.3f;clears[0].color.float32[1]=0.5f;
            clears[0].color.float32[2]=0.9f;clears[0].color.float32[3]=1.0f;
            clears[1].depthStencil={1.0f,0};

            VkRenderPassBeginInfo rp{};rp.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp.renderPass=sceneRenderPass_;rp.framebuffer=sceneFramebuffers_[imageIndex];
            rp.renderArea={{0,0},swapchainExtent_};
            rp.clearValueCount=static_cast<uint32_t>(clears.size());rp.pClearValues=clears.data();

            vkCmdBeginRenderPass(cmd,&rp,VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,scenePipeline_);

            VkViewport vp{0,0,(float)swapchainExtent_.width,(float)swapchainExtent_.height,0,1};
            vkCmdSetViewport(cmd,0,1,&vp);
            VkRect2D sc{{0,0},swapchainExtent_};vkCmdSetScissor(cmd,0,1,&sc);

            VkBuffer vb[]={vertexBuffer_};VkDeviceSize off[]={0};
            vkCmdBindVertexBuffers(cmd,0,1,vb,off);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,
                scenePipelineLayout_,0,1,&sceneDescSets_[currentFrame_],0,nullptr);
            vkCmdDraw(cmd,static_cast<uint32_t>(VERTICES.size()),1,0,0);

            vkCmdEndRenderPass(cmd);
        }

        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void updateSceneUBO(uint32_t frame)
    {
        static auto start=std::chrono::high_resolution_clock::now();
        float t=std::chrono::duration<float>(std::chrono::high_resolution_clock::now()-start).count();
        SceneUBO ubo{};
        ubo.model=glm::mat4(1.0f);
        ubo.view=glm::lookAt(glm::vec3(3.5f*std::cos(t*0.4f),3.0f,3.5f*std::sin(t*0.4f)),
            glm::vec3(0,0,0),glm::vec3(0,1,0));
        ubo.projection=glm::perspective(glm::radians(60.0f),
            (float)swapchainExtent_.width/swapchainExtent_.height,0.1f,20.0f);
        ubo.projection[1][1]*=-1;
        ubo.lightSpaceMatrix=getLightSpaceMatrix();
        ubo.lightDir=glm::vec4(lightDir_,0.0f);
        std::memcpy(sceneUBOMapped_[frame],&ubo,sizeof(ubo));
    }

    void drawFrame()
    {
        vkWaitForFences(device_,1,&inFlightFences_[currentFrame_],VK_TRUE,UINT64_MAX);
        uint32_t imgIdx=0;
        VkResult r=vkAcquireNextImageKHR(device_,swapchain_,UINT64_MAX,
            imageAvailableSems_[currentFrame_],VK_NULL_HANDLE,&imgIdx);
        if(r==VK_ERROR_OUT_OF_DATE_KHR){recreateSwapchain();return;}
        updateSceneUBO(currentFrame_);
        vkResetFences(device_,1,&inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_],0);
        recordCommandBuffer(commandBuffers_[currentFrame_],imgIdx);
        VkSemaphore ws[]={imageAvailableSems_[currentFrame_]};
        VkPipelineStageFlags wst[]={VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore ss[]={renderFinishedSems_[currentFrame_]};
        VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount=1;si.pWaitSemaphores=ws;si.pWaitDstStageMask=wst;
        si.commandBufferCount=1;si.pCommandBuffers=&commandBuffers_[currentFrame_];
        si.signalSemaphoreCount=1;si.pSignalSemaphores=ss;
        VK_CHECK(vkQueueSubmit(graphicsQueue_,1,&si,inFlightFences_[currentFrame_]));
        VkSwapchainKHR scs[]={swapchain_};
        VkPresentInfoKHR pi{};pi.sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount=1;pi.pWaitSemaphores=ss;pi.swapchainCount=1;pi.pSwapchains=scs;pi.pImageIndices=&imgIdx;
        r=vkQueuePresentKHR(presentQueue_,&pi);
        if(r==VK_ERROR_OUT_OF_DATE_KHR||r==VK_SUBOPTIMAL_KHR||resized_){resized_=false;recreateSwapchain();}
        currentFrame_=(currentFrame_+1)%MAX_FRAMES;
    }

    void mainLoop()
    {
        std::cout<<"🌑 Shadow Mapping（两趟渲染）运行中（ESC 退出）...\n";
        while(!glfwWindowShouldClose(window_)){
            glfwPollEvents();
            if(glfwGetKey(window_,GLFW_KEY_ESCAPE)==GLFW_PRESS)glfwSetWindowShouldClose(window_,GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    // ─── 辅助函数 ─────────────────────────────────────────────────────────────

    uint32_t findMemoryType(uint32_t f,VkMemoryPropertyFlags p)
    {
        VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(physicalDevice_,&mp);
        for(uint32_t i=0;i<mp.memoryTypeCount;++i)if((f&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&p)==p)return i;
        throw std::runtime_error("找不到合适内存");
    }
    void createBuffer(VkDeviceSize sz,VkBufferUsageFlags u,VkMemoryPropertyFlags p,VkBuffer&b,VkDeviceMemory&m)
    {
        VkBufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;ci.size=sz;ci.usage=u;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_,&ci,nullptr,&b));
        VkMemoryRequirements mr;vkGetBufferMemoryRequirements(device_,b,&mr);
        VkMemoryAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;ai.allocationSize=mr.size;ai.memoryTypeIndex=findMemoryType(mr.memoryTypeBits,p);
        VK_CHECK(vkAllocateMemory(device_,&ai,nullptr,&m));VK_CHECK(vkBindBufferMemory(device_,b,m,0));
    }
    VkFormat findSupportedFormat(const std::vector<VkFormat>&c,VkImageTiling t,VkFormatFeatureFlags f)
    {
        for(VkFormat fmt:c){VkFormatProperties p;vkGetPhysicalDeviceFormatProperties(physicalDevice_,fmt,&p);if(t==VK_IMAGE_TILING_OPTIMAL&&(p.optimalTilingFeatures&f)==f)return fmt;}
        throw std::runtime_error("找不到格式");
    }
    VkFormat findDepthFormat(){return findSupportedFormat({VK_FORMAT_D32_SFLOAT,VK_FORMAT_D32_SFLOAT_S8_UINT,VK_FORMAT_D24_UNORM_S8_UINT},VK_IMAGE_TILING_OPTIMAL,VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);}
    void createVertexBuffer()
    {
        VkDeviceSize sz=sizeof(VERTICES[0])*VERTICES.size();
        createBuffer(sz,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,vertexBuffer_,vertexMemory_);
        void*d=nullptr;vkMapMemory(device_,vertexMemory_,0,sz,0,&d);std::memcpy(d,VERTICES.data(),(size_t)sz);vkUnmapMemory(device_,vertexMemory_);
    }
    void createSceneUBOs()
    {
        VkDeviceSize sz=sizeof(SceneUBO);sceneUBOs_.resize(MAX_FRAMES);sceneUBOMemories_.resize(MAX_FRAMES);sceneUBOMapped_.resize(MAX_FRAMES);
        for(int i=0;i<MAX_FRAMES;++i){createBuffer(sz,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,sceneUBOs_[i],sceneUBOMemories_[i]);vkMapMemory(device_,sceneUBOMemories_[i],0,sz,0,&sceneUBOMapped_[i]);}
    }
    void createDepthResources()
    {
        VkImageCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;ci.imageType=VK_IMAGE_TYPE_2D;ci.extent={swapchainExtent_.width,swapchainExtent_.height,1};ci.mipLevels=1;ci.arrayLayers=1;ci.format=depthFormat_;ci.tiling=VK_IMAGE_TILING_OPTIMAL;ci.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;ci.usage=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;ci.samples=VK_SAMPLE_COUNT_1_BIT;
        VK_CHECK(vkCreateImage(device_,&ci,nullptr,&depthImage_));
        VkMemoryRequirements mr;vkGetImageMemoryRequirements(device_,depthImage_,&mr);
        VkMemoryAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;ai.allocationSize=mr.size;ai.memoryTypeIndex=findMemoryType(mr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device_,&ai,nullptr,&depthImageMemory_));VK_CHECK(vkBindImageMemory(device_,depthImage_,depthImageMemory_,0));
        VkImageViewCreateInfo vci{};vci.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;vci.image=depthImage_;vci.viewType=VK_IMAGE_VIEW_TYPE_2D;vci.format=depthFormat_;vci.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};
        VK_CHECK(vkCreateImageView(device_,&vci,nullptr,&depthImageView_));
    }
    void createSceneRenderPass()
    {
        VkAttachmentDescription ca{};ca.format=swapchainImageFormat_;ca.samples=VK_SAMPLE_COUNT_1_BIT;ca.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;ca.storeOp=VK_ATTACHMENT_STORE_OP_STORE;ca.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;ca.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;ca.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;ca.finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentDescription da{};da.format=depthFormat_;da.samples=VK_SAMPLE_COUNT_1_BIT;da.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;da.storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;da.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;da.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;da.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;da.finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference cr{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};VkAttachmentReference dr{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};sp.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sp.colorAttachmentCount=1;sp.pColorAttachments=&cr;sp.pDepthStencilAttachment=&dr;
        VkSubpassDependency dep{};dep.srcSubpass=VK_SUBPASS_EXTERNAL;dep.dstSubpass=0;dep.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;dep.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;dep.srcAccessMask=0;dep.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        std::array<VkAttachmentDescription,2> atts={ca,da};
        VkRenderPassCreateInfo rpi{};rpi.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;rpi.attachmentCount=static_cast<uint32_t>(atts.size());rpi.pAttachments=atts.data();rpi.subpassCount=1;rpi.pSubpasses=&sp;rpi.dependencyCount=1;rpi.pDependencies=&dep;
        VK_CHECK(vkCreateRenderPass(device_,&rpi,nullptr,&sceneRenderPass_));
    }
    void createSceneDescriptorSetLayout()
    {
        std::array<VkDescriptorSetLayoutBinding,2> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        bindings[1]={1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;ci.bindingCount=static_cast<uint32_t>(bindings.size());ci.pBindings=bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_,&ci,nullptr,&sceneSetLayout_));
    }
    void createScenePipeline()
    {
        VkShaderModule vert=createShaderModuleFromFile(device_,"shadow_scene.vert.spv");
        VkShaderModule frag=createShaderModuleFromFile(device_,"shadow_scene.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vert,"main",nullptr};
        stages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,frag,"main",nullptr};
        VkVertexInputBindingDescription bind{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription,3> attrs{};
        attrs[0]={0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,pos)};
        attrs[1]={1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,normal)};
        attrs[2]={2,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,color)};
        VkPipelineVertexInputStateCreateInfo vi{};vi.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;vi.vertexBindingDescriptionCount=1;vi.pVertexBindingDescriptions=&bind;vi.vertexAttributeDescriptionCount=static_cast<uint32_t>(attrs.size());vi.pVertexAttributeDescriptions=attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,nullptr,0,VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,nullptr,0,1,nullptr,1,nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};rs.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;rs.polygonMode=VK_POLYGON_MODE_FILL;rs.lineWidth=1.0f;rs.cullMode=VK_CULL_MODE_NONE;rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};ms.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};ds.sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;ds.depthTestEnable=VK_TRUE;ds.depthWriteEnable=VK_TRUE;ds.depthCompareOp=VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState cba{};cba.colorWriteMask=0xf;
        VkPipelineColorBlendStateCreateInfo cb{};cb.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;cb.attachmentCount=1;cb.pAttachments=&cba;
        std::vector<VkDynamicState> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};dynS.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;dynS.dynamicStateCount=static_cast<uint32_t>(dyn.size());dynS.pDynamicStates=dyn.data();
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.setLayoutCount=1;pli.pSetLayouts=&sceneSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_,&pli,nullptr,&scenePipelineLayout_));
        VkGraphicsPipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;pi.stageCount=2;pi.pStages=stages;pi.pVertexInputState=&vi;pi.pInputAssemblyState=&ia;pi.pViewportState=&vs;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pDepthStencilState=&ds;pi.pColorBlendState=&cb;pi.pDynamicState=&dynS;pi.layout=scenePipelineLayout_;pi.renderPass=sceneRenderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&scenePipeline_));
        vkDestroyShaderModule(device_,frag,nullptr);vkDestroyShaderModule(device_,vert,nullptr);
        std::cout<<"✅ 场景管线创建（Diffuse + PCF Shadow）\n";
    }
    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize,2> ps{};
        ps[0]={VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,static_cast<uint32_t>(MAX_FRAMES)};
        ps[1]={VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,static_cast<uint32_t>(MAX_FRAMES)};
        VkDescriptorPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;ci.poolSizeCount=static_cast<uint32_t>(ps.size());ci.pPoolSizes=ps.data();ci.maxSets=static_cast<uint32_t>(MAX_FRAMES);
        VK_CHECK(vkCreateDescriptorPool(device_,&ci,nullptr,&descriptorPool_));
    }
    void createSceneDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> lays(MAX_FRAMES,sceneSetLayout_);
        VkDescriptorSetAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;ai.descriptorPool=descriptorPool_;ai.descriptorSetCount=static_cast<uint32_t>(MAX_FRAMES);ai.pSetLayouts=lays.data();
        sceneDescSets_.resize(MAX_FRAMES);VK_CHECK(vkAllocateDescriptorSets(device_,&ai,sceneDescSets_.data()));
        for(int i=0;i<MAX_FRAMES;++i){
            VkDescriptorBufferInfo bi{};bi.buffer=sceneUBOs_[i];bi.offset=0;bi.range=sizeof(SceneUBO);
            VkDescriptorImageInfo ii{};ii.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;ii.imageView=shadowImageView_;ii.sampler=shadowSampler_;
            std::array<VkWriteDescriptorSet,2> ws{};
            ws[0]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,sceneDescSets_[i],0,0,1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,nullptr,&bi,nullptr};
            ws[1]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,sceneDescSets_[i],1,0,1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,&ii,nullptr,nullptr};
            vkUpdateDescriptorSets(device_,static_cast<uint32_t>(ws.size()),ws.data(),0,nullptr);
        }
    }
    void createSceneFramebuffers()
    {
        sceneFramebuffers_.resize(swapchainImageViews_.size());
        for(size_t i=0;i<swapchainImageViews_.size();++i){
            std::array<VkImageView,2> att={swapchainImageViews_[i],depthImageView_};
            VkFramebufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;ci.renderPass=sceneRenderPass_;ci.attachmentCount=static_cast<uint32_t>(att.size());ci.pAttachments=att.data();ci.width=swapchainExtent_.width;ci.height=swapchainExtent_.height;ci.layers=1;
            VK_CHECK(vkCreateFramebuffer(device_,&ci,nullptr,&sceneFramebuffers_[i]));
        }
    }
    void createInstance()
    {
        VkApplicationInfo ai{};ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;ai.apiVersion=VK_API_VERSION_1_3;
        auto exts=getRequiredInstanceExtensions();VkInstanceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ci.pApplicationInfo=&ai;ci.enabledExtensionCount=static_cast<uint32_t>(exts.size());ci.ppEnabledExtensionNames=exts.data();ci.flags|=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}
        VK_CHECK(vkCreateInstance(&ci,nullptr,&instance_));
    }
    void createSurface(){VK_CHECK(glfwCreateWindowSurface(instance_,window_,nullptr,&surface_));}
    void pickPhysicalDevice()
    {
        uint32_t c=0;vkEnumeratePhysicalDevices(instance_,&c,nullptr);std::vector<VkPhysicalDevice> devs(c);vkEnumeratePhysicalDevices(instance_,&c,devs.data());
        for(auto&d:devs)if(findQueueFamilies(d,surface_).isComplete()&&checkDeviceExtensionSupport(d)){physicalDevice_=d;break;}
        if(!physicalDevice_)throw std::runtime_error("无合适GPU");
        VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(physicalDevice_,&p);std::cout<<"✅ GPU: "<<p.deviceName<<"\n";
    }
    void createLogicalDevice()
    {
        queueIndices_=findQueueFamilies(physicalDevice_,surface_);
        std::set<uint32_t> fams={queueIndices_.graphicsFamily.value(),queueIndices_.presentFamily.value()};
        const float pri=1.0f;std::vector<VkDeviceQueueCreateInfo> qcis;
        for(uint32_t f:fams){VkDeviceQueueCreateInfo q{};q.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;q.queueFamilyIndex=f;q.queueCount=1;q.pQueuePriorities=&pri;qcis.push_back(q);}
        VkPhysicalDeviceFeatures feat{};feat.samplerAnisotropy=VK_TRUE;
        VkDeviceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;ci.queueCreateInfoCount=static_cast<uint32_t>(qcis.size());ci.pQueueCreateInfos=qcis.data();ci.pEnabledFeatures=&feat;ci.enabledExtensionCount=static_cast<uint32_t>(DEVICE_EXTENSIONS.size());ci.ppEnabledExtensionNames=DEVICE_EXTENSIONS.data();
        if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}
        VK_CHECK(vkCreateDevice(physicalDevice_,&ci,nullptr,&device_));
        vkGetDeviceQueue(device_,queueIndices_.graphicsFamily.value(),0,&graphicsQueue_);
        vkGetDeviceQueue(device_,queueIndices_.presentFamily.value(),0,&presentQueue_);
    }
    void createSwapchain()
    {
        auto sc=querySwapChainSupport(physicalDevice_,surface_);auto fmt=chooseSwapSurfaceFormat(sc.formats);auto mode=chooseSwapPresentMode(sc.presentModes);
        swapchainExtent_=chooseSwapExtent(sc.capabilities,window_);uint32_t n=sc.capabilities.minImageCount+1;if(sc.capabilities.maxImageCount>0)n=std::min(n,sc.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{};ci.sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;ci.surface=surface_;ci.minImageCount=n;ci.imageFormat=fmt.format;ci.imageColorSpace=fmt.colorSpace;ci.imageExtent=swapchainExtent_;ci.imageArrayLayers=1;ci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;ci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE;ci.preTransform=sc.capabilities.currentTransform;ci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;ci.presentMode=mode;ci.clipped=VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_,&ci,nullptr,&swapchain_));vkGetSwapchainImagesKHR(device_,swapchain_,&n,nullptr);swapchainImages_.resize(n);vkGetSwapchainImagesKHR(device_,swapchain_,&n,swapchainImages_.data());swapchainImageFormat_=fmt.format;
    }
    void createImageViews()
    {
        swapchainImageViews_.resize(swapchainImages_.size());
        for(size_t i=0;i<swapchainImages_.size();++i){
            VkImageViewCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;ci.image=swapchainImages_[i];ci.viewType=VK_IMAGE_VIEW_TYPE_2D;ci.format=swapchainImageFormat_;ci.components={VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY};ci.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            VK_CHECK(vkCreateImageView(device_,&ci,nullptr,&swapchainImageViews_[i]));
        }
    }
    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;ci.queueFamilyIndex=queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_,&ci,nullptr,&commandPool_));
    }
    void createCommandBuffers()
    {
        commandBuffers_.resize(MAX_FRAMES);VkCommandBufferAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;ai.commandPool=commandPool_;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=static_cast<uint32_t>(commandBuffers_.size());
        VK_CHECK(vkAllocateCommandBuffers(device_,&ai,commandBuffers_.data()));
    }
    void createSyncObjects()
    {
        imageAvailableSems_.resize(MAX_FRAMES);renderFinishedSems_.resize(MAX_FRAMES);inFlightFences_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo sCI{};sCI.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;VkFenceCreateInfo fCI{};fCI.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;fCI.flags=VK_FENCE_CREATE_SIGNALED_BIT;
        for(int i=0;i<MAX_FRAMES;++i){VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&imageAvailableSems_[i]));VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&renderFinishedSems_[i]));VK_CHECK(vkCreateFence(device_,&fCI,nullptr,&inFlightFences_[i]));}
    }
    void recreateSwapchain()
    {
        int w=0,h=0;glfwGetFramebufferSize(window_,&w,&h);while(!w||!h){glfwGetFramebufferSize(window_,&w,&h);glfwWaitEvents();}
        vkDeviceWaitIdle(device_);
        for(auto&fb:sceneFramebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);
        vkDestroyImageView(device_,depthImageView_,nullptr);vkDestroyImage(device_,depthImage_,nullptr);vkFreeMemory(device_,depthImageMemory_,nullptr);
        for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        createSwapchain();createImageViews();createDepthResources();createSceneFramebuffers();
    }
    void cleanup()
    {
        vkDestroySampler(device_,shadowSampler_,nullptr);vkDestroyImageView(device_,shadowImageView_,nullptr);vkDestroyImage(device_,shadowImage_,nullptr);vkFreeMemory(device_,shadowImageMemory_,nullptr);
        vkDestroyFramebuffer(device_,shadowFramebuffer_,nullptr);vkDestroyRenderPass(device_,shadowRenderPass_,nullptr);
        vkDestroyPipeline(device_,shadowPipeline_,nullptr);vkDestroyPipelineLayout(device_,shadowPipelineLayout_,nullptr);
        for(int i=0;i<MAX_FRAMES;++i){vkDestroyBuffer(device_,sceneUBOs_[i],nullptr);vkFreeMemory(device_,sceneUBOMemories_[i],nullptr);}
        vkDestroyDescriptorPool(device_,descriptorPool_,nullptr);vkDestroyDescriptorSetLayout(device_,sceneSetLayout_,nullptr);
        vkDestroyBuffer(device_,vertexBuffer_,nullptr);vkFreeMemory(device_,vertexMemory_,nullptr);
        for(int i=0;i<MAX_FRAMES;++i){vkDestroySemaphore(device_,imageAvailableSems_[i],nullptr);vkDestroySemaphore(device_,renderFinishedSems_[i],nullptr);vkDestroyFence(device_,inFlightFences_[i],nullptr);}
        vkDestroyCommandPool(device_,commandPool_,nullptr);
        for(auto&fb:sceneFramebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);
        vkDestroyImageView(device_,depthImageView_,nullptr);vkDestroyImage(device_,depthImage_,nullptr);vkFreeMemory(device_,depthImageMemory_,nullptr);
        vkDestroyPipeline(device_,scenePipeline_,nullptr);vkDestroyPipelineLayout(device_,scenePipelineLayout_,nullptr);vkDestroyRenderPass(device_,sceneRenderPass_,nullptr);
        for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        vkDestroyDevice(device_,nullptr);vkDestroySurfaceKHR(instance_,surface_,nullptr);vkDestroyInstance(instance_,nullptr);glfwDestroyWindow(window_);glfwTerminate();
        std::cout<<"✅ 清理完成。\n";
    }
};

int main()
{
    std::cout<<"═══════════════════════════════════════════════════════════════════\n";
    std::cout<<" 第18章：Shadow Mapping（阴影映射）\n";
    std::cout<<"\n";
    std::cout<<" 两趟渲染：\n";
    std::cout<<"   Pass1: 光源视角 → Shadow Map（只写深度）\n";
    std::cout<<"   Pass2: 相机视角 → 每像素与 Shadow Map 比较 → PCF 软阴影\n";
    std::cout<<"\n";
    std::cout<<" 关键技术：\n";
    std::cout<<"   • VkRenderPass（仅深度）→ Shadow Map\n";
    std::cout<<"   • 深度偏移（depthBiasSlopeFactor）防止 shadow acne\n";
    std::cout<<"   • PCF 3×3 采样产生软阴影\n";
    std::cout<<"   • Push Constants 传递 lightSpaceMatrix\n";
    std::cout<<"═══════════════════════════════════════════════════════════════════\n\n";
    Ch18App app;
    try{app.run();}catch(const std::exception&e){std::cerr<<"❌ "<<e.what()<<"\n";return 1;}
    return 0;
}
