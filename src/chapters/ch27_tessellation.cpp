/**
 * @file ch27_tessellation.cpp
 * @brief 第27章：曲面细分着色器（Tessellation）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【什么是曲面细分？】
 *
 *  将粗糙的几何体（低面数网格）在 GPU 上自动细分成更多三角形，
 *  近处高精度，远处低精度，平衡质量和性能。
 *
 * 【两个新着色器阶段】
 *
 *  传统管线：  Vertex → Rasterization → Fragment
 *  细分管线：  Vertex → TCS → Tessellator（固定功能）→ TES → Fragment
 *
 *  TCS（Tessellation Control Shader / Hull Shader）：
 *    - 决定每条边分成几段（TessLevel）
 *    - 处理控制点（Control Points）
 *    - 类似 Task Shader，每个 patch 运行一次
 *
 *  固定功能 Tessellator：
 *    - 根据 TessLevel 生成细分后的顶点坐标（重心坐标）
 *    - 程序员不能修改
 *
 *  TES（Tessellation Evaluation Shader / Domain Shader）：
 *    - 为每个细分后的顶点计算实际位置
 *    - 可以实现曲面（Bezier/PN 三角形）、地形置换等
 *    - 类似顶点着色器，每个细分顶点调用一次
 *
 * 【patch 拓扑】
 *
 *  三角形 patch（本章使用）：
 *    - 3 个控制点，gl_TessCoord = 重心坐标 (u,v,w)
 *    - TessLevelOuter[0/1/2]：三角形三条边各分几段
 *    - TessLevelInner[0]：内部细分密度
 *
 *  四边形 patch：
 *    - 4 个控制点，gl_TessCoord = (u,v)
 *    - 用于地形、贝塞尔曲面
 *
 * 【置换贴图（Displacement Mapping）】
 *
 *  TES 中沿法线方向位移顶点，实现凹凸地表：
 *    pos += normal * displacementMap.sample(uv)
 *
 * 【本章示例】
 *
 *  球面细分 + 正弦波置换：
 *  - 基础网格：一个 20-面体（粗糙球）
 *  - TessLevel 由滑块控制（1-64）
 *  - TES：沿法线方向添加正弦波形变形
 *  - 按 +/- 键调节细分级别，观察效果变化
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

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int      MAX_FRAMES = 2;

struct TessVertex { glm::vec3 pos; glm::vec3 normal; glm::vec3 color; };
struct TessPushConstants { glm::mat4 mvp; float tessLevel; float time; };

// ─── 简单的球体网格（正二十面体近似） ────────────────────────────────────────

static std::vector<TessVertex> buildSphere(int subdivisions = 2)
{
    // 正二十面体 12 个顶点（归一化到单位球）
    const float t = (1.0f + sqrtf(5.0f)) / 2.0f;
    std::vector<glm::vec3> verts = {
        glm::normalize(glm::vec3(-1, t, 0)), glm::normalize(glm::vec3(1, t, 0)),
        glm::normalize(glm::vec3(-1,-t, 0)), glm::normalize(glm::vec3(1,-t, 0)),
        glm::normalize(glm::vec3(0,-1, t)), glm::normalize(glm::vec3(0, 1, t)),
        glm::normalize(glm::vec3(0,-1,-t)), glm::normalize(glm::vec3(0, 1,-t)),
        glm::normalize(glm::vec3(t, 0,-1)), glm::normalize(glm::vec3(t, 0, 1)),
        glm::normalize(glm::vec3(-t,0,-1)), glm::normalize(glm::vec3(-t,0, 1)),
    };
    // 正二十面体 20 个面（索引）
    std::vector<std::array<int,3>> faces = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1},
    };

    std::vector<TessVertex> result;
    for (auto& f : faces) {
        for (int i : f) {
            glm::vec3 p = verts[i];
            glm::vec3 c = p * 0.5f + 0.5f;   // 法线映射到颜色空间
            result.push_back({p * 0.7f, p, c});
        }
    }
    return result;
}

class Ch27App {
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow*      window_         = nullptr;
    VkInstance       instance_       = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_        = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          graphicsQueue_  = VK_NULL_HANDLE;
    VkQueue          presentQueue_   = VK_NULL_HANDLE;
    VkSwapchainKHR   swapchain_      = VK_NULL_HANDLE;
    VkRenderPass     renderPass_     = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline       pipeline_       = VK_NULL_HANDLE;   // 实体模式
    VkPipeline       wireframePipe_  = VK_NULL_HANDLE;   // 线框模式（观察细分效果）
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;
    VkBuffer         vertexBuffer_   = VK_NULL_HANDLE;
    VkDeviceMemory   vertexMemory_   = VK_NULL_HANDLE;

    std::vector<TessVertex>  vertices_;
    uint32_t                 vertexCount_ = 0;
    float                    tessLevel_   = 4.0f;   // 细分级别（交互调节）
    bool                     wireframe_   = false;

    std::vector<VkImage>         swapchainImages_;
    std::vector<VkImageView>     swapchainImageViews_;
    std::vector<VkFramebuffer>   framebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkFormat                     swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D                   swapchainExtent_{};
    QueueFamilyIndices           queueIndices_;

    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence>     inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool     resized_      = false;

    void initWindow()
    {
        glfwInit(); glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT,
            "Ch27 - 曲面细分（+/-调整精度 / W切换线框）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_,
            [](GLFWwindow*w,int,int){
                reinterpret_cast<Ch27App*>(glfwGetWindowUserPointer(w))->resized_=true;
            });
        glfwSetKeyCallback(window_, [](GLFWwindow* w, int key, int, int action, int) {
            if (action != GLFW_PRESS) return;
            auto* app = reinterpret_cast<Ch27App*>(glfwGetWindowUserPointer(w));
            if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD)
                app->tessLevel_ = std::min(app->tessLevel_ + 2.0f, 64.0f);
            else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT)
                app->tessLevel_ = std::max(app->tessLevel_ - 2.0f, 1.0f);
            else if (key == GLFW_KEY_W)
                app->wireframe_ = !app->wireframe_;
            std::cout << "TessLevel=" << app->tessLevel_
                      << "  Wireframe=" << app->wireframe_ << "\n";
        });
    }

    void initVulkan()
    {
        createInstance(); createSurface(); pickPhysicalDevice();
        createLogicalDeviceWithTessFeature();
        createSwapchain(); createImageViews(); createRenderPass();
        createTessellationPipelines();
        createFramebuffers(); createCommandPool();
        vertices_ = buildSphere(2);
        vertexCount_ = static_cast<uint32_t>(vertices_.size());
        createVertexBuffer();
        createCommandBuffers(); createSyncObjects();
        std::cout << "\n✅ 曲面细分初始化完成！\n";
        std::cout << "📐 基础网格：" << vertexCount_ / 3 << " 个三角形 patch\n";
        std::cout << "🔑 控制键：+/- 调整细分级别 | W 切换线框模式\n";
    }

    void createLogicalDeviceWithTessFeature()
    {
        queueIndices_=findQueueFamilies(physicalDevice_,surface_);
        std::set<uint32_t> fams={queueIndices_.graphicsFamily.value(),queueIndices_.presentFamily.value()};
        const float pri=1.0f;std::vector<VkDeviceQueueCreateInfo> qcis;
        for(uint32_t f:fams){VkDeviceQueueCreateInfo q{};q.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;q.queueFamilyIndex=f;q.queueCount=1;q.pQueuePriorities=&pri;qcis.push_back(q);}
        VkPhysicalDeviceFeatures feat{};
        feat.tessellationShader = VK_TRUE;   // ← 必须启用！
        feat.fillModeNonSolid   = VK_TRUE;   // ← 线框模式需要
        feat.samplerAnisotropy  = VK_TRUE;
        VkDeviceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;ci.queueCreateInfoCount=static_cast<uint32_t>(qcis.size());ci.pQueueCreateInfos=qcis.data();ci.pEnabledFeatures=&feat;ci.enabledExtensionCount=static_cast<uint32_t>(DEVICE_EXTENSIONS.size());ci.ppEnabledExtensionNames=DEVICE_EXTENSIONS.data();
        if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}
        VK_CHECK(vkCreateDevice(physicalDevice_,&ci,nullptr,&device_));
        vkGetDeviceQueue(device_,queueIndices_.graphicsFamily.value(),0,&graphicsQueue_);
        vkGetDeviceQueue(device_,queueIndices_.presentFamily.value(),0,&presentQueue_);
        std::cout<<"✅ 逻辑设备已创建（tessellationShader=true）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 创建细分管线（含 TCS + TES 阶段）
    // ═══════════════════════════════════════════════════════════════════════

    void createTessellationPipelines()
    {
        VkShaderModule vert = createShaderModuleFromFile(device_, "tess.vert.spv");
        VkShaderModule tcs  = createShaderModuleFromFile(device_, "tess.tesc.spv");
        VkShaderModule tes  = createShaderModuleFromFile(device_, "tess.tese.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "tess.frag.spv");

        // ── 4 个着色器阶段（TCS + TES 是新增的！） ──────────────────────────
        VkPipelineShaderStageCreateInfo stages[4]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,
                     VK_SHADER_STAGE_VERTEX_BIT,                  vert,"main",nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,
                     VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,    tcs, "main",nullptr}; // TCS
        stages[2] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,
                     VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, tes, "main",nullptr}; // TES
        stages[3] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,
                     VK_SHADER_STAGE_FRAGMENT_BIT,                frag,"main",nullptr};

        VkVertexInputBindingDescription bind{0,sizeof(TessVertex),VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription,3> attrs{};
        attrs[0]={0,0,VK_FORMAT_R32G32B32_SFLOAT,0};
        attrs[1]={1,0,VK_FORMAT_R32G32B32_SFLOAT,12};
        attrs[2]={2,0,VK_FORMAT_R32G32B32_SFLOAT,24};
        VkPipelineVertexInputStateCreateInfo vi{};vi.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;vi.vertexBindingDescriptionCount=1;vi.pVertexBindingDescriptions=&bind;vi.vertexAttributeDescriptionCount=3;vi.pVertexAttributeDescriptions=attrs.data();

        // ── Patch List：告诉管线每个 Patch 有几个控制点 ──────────────────────
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;  // ← 细分管线必须用 PATCH_LIST

        // ── 细分状态：每个 Patch 的控制点数（TCS 的 layout(vertices=3)） ──
        VkPipelineTessellationStateCreateInfo tessState{};
        tessState.sType              = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
        tessState.patchControlPoints = 3;   // 三角形：3 个控制点/patch

        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,nullptr,0,1,nullptr,1,nullptr};

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.lineWidth   = 1.0f;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.polygonMode = VK_POLYGON_MODE_FILL;  // 实体模式

        VkPipelineMultisampleStateCreateInfo ms{};ms.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};cba.colorWriteMask=0xf;
        VkPipelineColorBlendStateCreateInfo cb{};cb.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;cb.attachmentCount=1;cb.pAttachments=&cba;
        std::vector<VkDynamicState> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};dynS.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;dynS.dynamicStateCount=static_cast<uint32_t>(dyn.size());dynS.pDynamicStates=dyn.data();

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                             VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        pcRange.offset = 0; pcRange.size = sizeof(TessPushConstants);
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_,&pli,nullptr,&pipelineLayout_));

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType             = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount        = 4;    // ← 4 个阶段！
        pi.pStages           = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pTessellationState  = &tessState;  // ← 细分状态
        pi.pViewportState    = &vs;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pColorBlendState  = &cb;
        pi.pDynamicState     = &dynS;
        pi.layout            = pipelineLayout_;
        pi.renderPass        = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&pipeline_));

        // 线框管线（同样是细分，但 polygonMode = LINE）
        rs.polygonMode = VK_POLYGON_MODE_LINE;
        VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&wireframePipe_));

        vkDestroyShaderModule(device_,frag,nullptr);vkDestroyShaderModule(device_,tes,nullptr);
        vkDestroyShaderModule(device_,tcs,nullptr); vkDestroyShaderModule(device_,vert,nullptr);

        std::cout<<"✅ 细分管线创建成功（含 TCS + TES 阶段）\n";
        std::cout<<"   patchControlPoints=3（三角形细分）\n";
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
    {
        static auto start = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now() - start).count();

        VkCommandBufferBeginInfo bi{};bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd,&bi));

        VkClearValue clear{};clear.color.float32[0]=0.02f;clear.color.float32[2]=0.05f;clear.color.float32[3]=1.0f;
        VkRenderPassBeginInfo rp{};rp.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;rp.renderPass=renderPass_;rp.framebuffer=framebuffers_[imageIndex];rp.renderArea={{0,0},swapchainExtent_};rp.clearValueCount=1;rp.pClearValues=&clear;
        vkCmdBeginRenderPass(cmd,&rp,VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            wireframe_ ? wireframePipe_ : pipeline_);
        VkViewport vp{0,0,(float)swapchainExtent_.width,(float)swapchainExtent_.height,0,1};
        vkCmdSetViewport(cmd,0,1,&vp);
        VkRect2D sc{{0,0},swapchainExtent_};vkCmdSetScissor(cmd,0,1,&sc);
        VkBuffer vb[]={vertexBuffer_};VkDeviceSize off[]={0};
        vkCmdBindVertexBuffers(cmd,0,1,vb,off);

        glm::mat4 proj = glm::perspective(glm::radians(60.0f),
            (float)swapchainExtent_.width / swapchainExtent_.height, 0.1f, 10.0f);
        proj[1][1] *= -1;
        glm::mat4 view = glm::lookAt(glm::vec3(0,0,2.5f), glm::vec3(0), glm::vec3(0,1,0));
        glm::mat4 model = glm::rotate(glm::mat4(1.0f), time * 0.5f, glm::vec3(0,1,0));

        TessPushConstants pc{};
        pc.mvp       = proj * view * model;
        pc.tessLevel = tessLevel_;
        pc.time      = time;
        vkCmdPushConstants(cmd, pipelineLayout_,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            0, sizeof(pc), &pc);

        // vkCmdDraw：仍然使用 vertexCount 个顶点
        // 管线会将每 3 个顶点组成一个 Patch，交给 TCS 处理
        vkCmdDraw(cmd, vertexCount_, 1, 0, 0);

        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame()
    {
        vkWaitForFences(device_,1,&inFlightFences_[currentFrame_],VK_TRUE,UINT64_MAX);
        uint32_t imgIdx=0;
        VkResult r=vkAcquireNextImageKHR(device_,swapchain_,UINT64_MAX,
            imageAvailableSems_[currentFrame_],VK_NULL_HANDLE,&imgIdx);
        if(r==VK_ERROR_OUT_OF_DATE_KHR){recreateSwapchain();return;}
        vkResetFences(device_,1,&inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_],0);
        recordCommandBuffer(commandBuffers_[currentFrame_],imgIdx);
        VkSemaphore ws[]={imageAvailableSems_[currentFrame_]};
        VkPipelineStageFlags wst[]={VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore ss[]={renderFinishedSems_[currentFrame_]};
        VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;si.waitSemaphoreCount=1;si.pWaitSemaphores=ws;si.pWaitDstStageMask=wst;si.commandBufferCount=1;si.pCommandBuffers=&commandBuffers_[currentFrame_];si.signalSemaphoreCount=1;si.pSignalSemaphores=ss;
        VK_CHECK(vkQueueSubmit(graphicsQueue_,1,&si,inFlightFences_[currentFrame_]));
        VkSwapchainKHR scs[]={swapchain_};VkPresentInfoKHR pi{};pi.sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;pi.waitSemaphoreCount=1;pi.pWaitSemaphores=ss;pi.swapchainCount=1;pi.pSwapchains=scs;pi.pImageIndices=&imgIdx;
        r=vkQueuePresentKHR(presentQueue_,&pi);
        if(r==VK_ERROR_OUT_OF_DATE_KHR||r==VK_SUBOPTIMAL_KHR||resized_){resized_=false;recreateSwapchain();}
        currentFrame_=(currentFrame_+1)%MAX_FRAMES;
    }

    void mainLoop()
    {
        std::cout<<"🔮 旋转球体（曲面细分）| +/-调节精度 | W切换线框 | ESC退出\n";
        while(!glfwWindowShouldClose(window_)){
            glfwPollEvents();
            if(glfwGetKey(window_,GLFW_KEY_ESCAPE)==GLFW_PRESS)
                glfwSetWindowShouldClose(window_,GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    uint32_t findMemoryType(uint32_t f,VkMemoryPropertyFlags p){VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(physicalDevice_,&mp);for(uint32_t i=0;i<mp.memoryTypeCount;++i)if((f&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&p)==p)return i;throw std::runtime_error("找不到内存类型");}
    void createBuffer(VkDeviceSize sz,VkBufferUsageFlags u,VkMemoryPropertyFlags p,VkBuffer&b,VkDeviceMemory&m){VkBufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;ci.size=sz;ci.usage=u;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;VK_CHECK(vkCreateBuffer(device_,&ci,nullptr,&b));VkMemoryRequirements mr;vkGetBufferMemoryRequirements(device_,b,&mr);VkMemoryAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;ai.allocationSize=mr.size;ai.memoryTypeIndex=findMemoryType(mr.memoryTypeBits,p);VK_CHECK(vkAllocateMemory(device_,&ai,nullptr,&m));VK_CHECK(vkBindBufferMemory(device_,b,m,0));}
    void createVertexBuffer(){VkDeviceSize sz=sizeof(TessVertex)*vertices_.size();createBuffer(sz,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,vertexBuffer_,vertexMemory_);void*d=nullptr;vkMapMemory(device_,vertexMemory_,0,sz,0,&d);std::memcpy(d,vertices_.data(),(size_t)sz);vkUnmapMemory(device_,vertexMemory_);}
    void createInstance(){VkApplicationInfo ai{};ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;ai.apiVersion=VK_API_VERSION_1_3;auto exts=getRequiredInstanceExtensions();VkInstanceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ci.pApplicationInfo=&ai;ci.enabledExtensionCount=static_cast<uint32_t>(exts.size());ci.ppEnabledExtensionNames=exts.data();ci.flags|=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}VK_CHECK(vkCreateInstance(&ci,nullptr,&instance_));}
    void createSurface(){VK_CHECK(glfwCreateWindowSurface(instance_,window_,nullptr,&surface_));}
    void pickPhysicalDevice(){uint32_t c=0;vkEnumeratePhysicalDevices(instance_,&c,nullptr);std::vector<VkPhysicalDevice> devs(c);vkEnumeratePhysicalDevices(instance_,&c,devs.data());for(auto&d:devs)if(findQueueFamilies(d,surface_).isComplete()&&checkDeviceExtensionSupport(d)){physicalDevice_=d;break;}if(!physicalDevice_)throw std::runtime_error("无合适GPU");VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(physicalDevice_,&p);std::cout<<"✅ GPU: "<<p.deviceName<<"\n";}
    void createSwapchain(){auto sc=querySwapChainSupport(physicalDevice_,surface_);auto fmt=chooseSwapSurfaceFormat(sc.formats);auto mode=chooseSwapPresentMode(sc.presentModes);swapchainExtent_=chooseSwapExtent(sc.capabilities,window_);uint32_t n=sc.capabilities.minImageCount+1;if(sc.capabilities.maxImageCount>0)n=std::min(n,sc.capabilities.maxImageCount);VkSwapchainCreateInfoKHR ci{};ci.sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;ci.surface=surface_;ci.minImageCount=n;ci.imageFormat=fmt.format;ci.imageColorSpace=fmt.colorSpace;ci.imageExtent=swapchainExtent_;ci.imageArrayLayers=1;ci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;ci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE;ci.preTransform=sc.capabilities.currentTransform;ci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;ci.presentMode=mode;ci.clipped=VK_TRUE;VK_CHECK(vkCreateSwapchainKHR(device_,&ci,nullptr,&swapchain_));vkGetSwapchainImagesKHR(device_,swapchain_,&n,nullptr);swapchainImages_.resize(n);vkGetSwapchainImagesKHR(device_,swapchain_,&n,swapchainImages_.data());swapchainImageFormat_=fmt.format;}
    void createImageViews(){swapchainImageViews_.resize(swapchainImages_.size());for(size_t i=0;i<swapchainImages_.size();++i){VkImageViewCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;ci.image=swapchainImages_[i];ci.viewType=VK_IMAGE_VIEW_TYPE_2D;ci.format=swapchainImageFormat_;ci.components={VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY};ci.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};VK_CHECK(vkCreateImageView(device_,&ci,nullptr,&swapchainImageViews_[i]));}}
    void createRenderPass(){VkAttachmentDescription ca{};ca.format=swapchainImageFormat_;ca.samples=VK_SAMPLE_COUNT_1_BIT;ca.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;ca.storeOp=VK_ATTACHMENT_STORE_OP_STORE;ca.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;ca.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;ca.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;ca.finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;VkAttachmentReference cr{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};VkSubpassDescription sp{};sp.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sp.colorAttachmentCount=1;sp.pColorAttachments=&cr;VkSubpassDependency dep{};dep.srcSubpass=VK_SUBPASS_EXTERNAL;dep.dstSubpass=0;dep.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;dep.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;dep.srcAccessMask=0;dep.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;VkRenderPassCreateInfo rpi{};rpi.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;rpi.attachmentCount=1;rpi.pAttachments=&ca;rpi.subpassCount=1;rpi.pSubpasses=&sp;rpi.dependencyCount=1;rpi.pDependencies=&dep;VK_CHECK(vkCreateRenderPass(device_,&rpi,nullptr,&renderPass_));}
    void createFramebuffers(){framebuffers_.resize(swapchainImageViews_.size());for(size_t i=0;i<swapchainImageViews_.size();++i){VkImageView att[]={swapchainImageViews_[i]};VkFramebufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;ci.renderPass=renderPass_;ci.attachmentCount=1;ci.pAttachments=att;ci.width=swapchainExtent_.width;ci.height=swapchainExtent_.height;ci.layers=1;VK_CHECK(vkCreateFramebuffer(device_,&ci,nullptr,&framebuffers_[i]));}}
    void createCommandPool(){VkCommandPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;ci.queueFamilyIndex=queueIndices_.graphicsFamily.value();VK_CHECK(vkCreateCommandPool(device_,&ci,nullptr,&commandPool_));}
    void createCommandBuffers(){commandBuffers_.resize(MAX_FRAMES);VkCommandBufferAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;ai.commandPool=commandPool_;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=static_cast<uint32_t>(commandBuffers_.size());VK_CHECK(vkAllocateCommandBuffers(device_,&ai,commandBuffers_.data()));}
    void createSyncObjects(){imageAvailableSems_.resize(MAX_FRAMES);renderFinishedSems_.resize(MAX_FRAMES);inFlightFences_.resize(MAX_FRAMES);VkSemaphoreCreateInfo sCI{};sCI.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;VkFenceCreateInfo fCI{};fCI.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;fCI.flags=VK_FENCE_CREATE_SIGNALED_BIT;for(int i=0;i<MAX_FRAMES;++i){VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&imageAvailableSems_[i]));VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&renderFinishedSems_[i]));VK_CHECK(vkCreateFence(device_,&fCI,nullptr,&inFlightFences_[i]));}}
    void recreateSwapchain(){int w=0,h=0;glfwGetFramebufferSize(window_,&w,&h);while(!w||!h){glfwGetFramebufferSize(window_,&w,&h);glfwWaitEvents();}vkDeviceWaitIdle(device_);for(auto&fb:framebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);vkDestroySwapchainKHR(device_,swapchain_,nullptr);createSwapchain();createImageViews();createFramebuffers();}
    void cleanup()
    {
        vkDestroyBuffer(device_,vertexBuffer_,nullptr);vkFreeMemory(device_,vertexMemory_,nullptr);
        for(int i=0;i<MAX_FRAMES;++i){vkDestroySemaphore(device_,imageAvailableSems_[i],nullptr);vkDestroySemaphore(device_,renderFinishedSems_[i],nullptr);vkDestroyFence(device_,inFlightFences_[i],nullptr);}
        vkDestroyCommandPool(device_,commandPool_,nullptr);for(auto&fb:framebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);
        vkDestroyPipeline(device_,wireframePipe_,nullptr);vkDestroyPipeline(device_,pipeline_,nullptr);
        vkDestroyPipelineLayout(device_,pipelineLayout_,nullptr);vkDestroyRenderPass(device_,renderPass_,nullptr);
        for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        vkDestroyDevice(device_,nullptr);vkDestroySurfaceKHR(instance_,surface_,nullptr);vkDestroyInstance(instance_,nullptr);glfwDestroyWindow(window_);glfwTerminate();
        std::cout<<"✅ 清理完成。\n";
    }
};

int main()
{
    std::cout<<"═══════════════════════════════════════════════════════════════════\n";
    std::cout<<" 第27章：曲面细分着色器（Tessellation Control + Evaluation）\n";
    std::cout<<"\n";
    std::cout<<" 管线：Vertex → TCS → [固定Tessellator] → TES → Fragment\n";
    std::cout<<"\n";
    std::cout<<" 关键点：\n";
    std::cout<<"   • topology = PATCH_LIST（不是 TRIANGLE_LIST！）\n";
    std::cout<<"   • patchControlPoints = 3（三角形patch）\n";
    std::cout<<"   • tessellationShader = VK_TRUE（设备特性）\n";
    std::cout<<"   • TCS：设置 gl_TessLevel，传递控制点\n";
    std::cout<<"   • TES：根据 gl_TessCoord 重心坐标计算最终顶点位置\n";
    std::cout<<"═══════════════════════════════════════════════════════════════════\n\n";
    Ch27App app;
    try{app.run();}catch(const std::exception&e){std::cerr<<"❌ "<<e.what()<<"\n";return 1;}
    return 0;
}
