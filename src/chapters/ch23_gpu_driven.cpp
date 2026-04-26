/**
 * @file ch23_gpu_driven.cpp
 * @brief 第23章：GPU-Driven Rendering（间接绘制 + GPU 视锥体剔除）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【CPU-Driven vs GPU-Driven Rendering】
 *
 *  CPU-Driven（传统方式）：
 *    CPU 遍历所有物体 → 视锥体剔除 → 对可见物体发出 draw call
 *    问题：大场景 (100k+ 物体) CPU 成为瓶颈
 *    问题：每帧从 GPU 读取物体列表 → PCIe 带宽浪费
 *
 *  GPU-Driven Rendering：
 *    1. CPU 将所有物体数据上传到 GPU SSBO（一次性）
 *    2. Compute Shader 在 GPU 上并行执行视锥体剔除
 *    3. 填充 Indirect Draw Buffer（VkDrawIndirectCommand 数组）
 *    4. 一次 vkCmdDrawIndirect 命令让 GPU 根据 Indirect Buffer 绘制
 *
 *  好处：
 *    ✅ 剔除在 GPU 上并行 → 极快（100k 物体 < 1ms）
 *    ✅ CPU 只发一次命令 → Draw Call 极少
 *    ✅ 无 GPU→CPU 数据回传 → 零等待
 *
 * 【间接绘制（Indirect Drawing）】
 *
 *  vkCmdDrawIndirect(cmd, indirectBuffer, offset, drawCount, stride)
 *    → GPU 从 indirectBuffer 读取绘制参数（不依赖 CPU！）
 *
 *  VkDrawIndirectCommand {
 *      uint32_t vertexCount;    // GPU 写入的绘制参数
 *      uint32_t instanceCount;  // = 0 表示剔除（不绘制）
 *      uint32_t firstVertex;
 *      uint32_t firstInstance;
 *  }
 *
 * 【完整 GPU-Driven 渲染帧流程】
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  Compute Pass：                                         │
 *  │    gpu_cull.comp                                        │
 *  │    - 并行遍历所有物体的包围球                            │
 *  │    - 视锥体测试：6平面法                                 │
 *  │    - 写入 IndirectCommands（可见=1，剔除=0）              │
 *  │                                                         │
 *  │  Pipeline Barrier（确保 Compute 写完 Indirect Buffer）   │
 *  │                                                         │
 *  │  Graphics Pass：                                        │
 *  │    vkCmdDrawIndirect(indirectBuffer, 0, N, stride)      │
 *  │    → GPU 读取 Indirect Buffer，自动跳过 instanceCount=0  │
 *  └─────────────────────────────────────────────────────────┘
 *
 * 【Multi-Draw Indirect（MDI）扩展优化】
 *
 *  VK_EXT_multi_draw：一次命令绘制多个不同网格
 *  vkCmdDrawMultiEXT(count, drawInfos, stride)
 *
 * 【本章示例】
 *  - 场景：2048 个随机大小的物体（包围球测试）
 *  - GPU 剔除计算着色器（gpu_cull.comp）
 *  - 间接绘制展示可见物体数量统计
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
#include <random>
#include <set>
#include <stdexcept>

constexpr uint32_t WIDTH      = 800;
constexpr uint32_t HEIGHT     = 600;
constexpr int      MAX_FRAMES = 2;
constexpr uint32_t N_OBJECTS  = 2048;   // 场景物体数

struct alignas(16) DrawObject {
    glm::vec4 boundingSphere;  // xyz=中心, w=半径
    uint32_t  vertexCount;
    uint32_t  firstVertex;
    uint32_t  padding[2];
};

struct VkDrawIndirectCommand_ {   // 对应 VkDrawIndirectCommand
    uint32_t vertexCount;
    uint32_t instanceCount;  // 0=剔除
    uint32_t firstVertex;
    uint32_t firstInstance;
};

struct CullPushConstants {
    glm::mat4 viewProjection;
    glm::vec4 frustumPlanes[6];  // 视锥体 6 个平面
    uint32_t  objectCount;
    float     padding[3];
};

struct ScenePushConstants {
    glm::mat4 viewProjection;
    float     time;
};

struct SceneVertex { glm::vec3 pos; glm::vec3 color; };

class Ch23App {
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
    VkQueue          computeQueue_   = VK_NULL_HANDLE;
    VkSwapchainKHR   swapchain_      = VK_NULL_HANDLE;
    VkRenderPass     renderPass_     = VK_NULL_HANDLE;
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;

    // ─── Compute Cull 管线 ─────────────────────────────────────────────────
    VkDescriptorSetLayout cullSetLayout_   = VK_NULL_HANDLE;
    VkPipelineLayout      cullPipeLayout_  = VK_NULL_HANDLE;
    VkPipeline            cullPipeline_    = VK_NULL_HANDLE;
    VkDescriptorPool      cullDescPool_    = VK_NULL_HANDLE;
    VkDescriptorSet       cullDescSet_     = VK_NULL_HANDLE;

    // ─── 图形管线 ──────────────────────────────────────────────────────────
    VkPipelineLayout      scenePipeLayout_  = VK_NULL_HANDLE;
    VkPipeline            scenePipeline_    = VK_NULL_HANDLE;

    // ─── 缓冲 ──────────────────────────────────────────────────────────────
    VkBuffer       objectBuffer_    = VK_NULL_HANDLE;  ///< 物体包围球数据
    VkDeviceMemory objectMemory_    = VK_NULL_HANDLE;
    VkBuffer       indirectBuffer_  = VK_NULL_HANDLE;  ///< 间接绘制命令缓冲
    VkDeviceMemory indirectMemory_  = VK_NULL_HANDLE;
    VkBuffer       countBuffer_     = VK_NULL_HANDLE;  ///< 可见物体计数
    VkDeviceMemory countMemory_     = VK_NULL_HANDLE;
    VkBuffer       vertexBuffer_    = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_    = VK_NULL_HANDLE;

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

    std::vector<DrawObject> sceneObjects_;

    void initWindow()
    {
        glfwInit(); glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT,
            "Ch23 - GPU-Driven Rendering（2048物体，GPU视锥剔除）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_,
            [](GLFWwindow*w,int,int){
                reinterpret_cast<Ch23App*>(glfwGetWindowUserPointer(w))->resized_=true;
            });
    }

    void initVulkan()
    {
        createInstance(); createSurface(); pickPhysicalDevice(); createLogicalDevice();
        createSwapchain(); createImageViews(); createRenderPass();
        createCullDescriptorSetLayout();
        createCullPipeline();    // Compute: GPU 视锥体剔除
        createScenePipeline();   // Graphics: 间接绘制
        createFramebuffers(); createCommandPool();
        generateSceneData();     // 随机生成 N_OBJECTS 个物体
        createBuffers();         // 物体缓冲 + 间接缓冲
        createCullDescriptorSet();
        createCommandBuffers(); createSyncObjects();
        std::cout << "\n✅ GPU-Driven Rendering 初始化完成！\n";
        std::cout << "🏗️  场景物体：" << N_OBJECTS << " 个\n";
        std::cout << "🔧 每帧流程：\n";
        std::cout << "   1. Compute Shader 视锥体剔除（GPU）\n";
        std::cout << "   2. Pipeline Barrier（同步 Indirect Buffer）\n";
        std::cout << "   3. vkCmdDrawIndirect（一次命令绘制所有可见物体）\n";
    }

    void generateSceneData()
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> distXZ(-15.0f, 15.0f);
        std::uniform_real_distribution<float> distY(-3.0f, 3.0f);
        std::uniform_real_distribution<float> distR(0.1f, 0.8f);

        sceneObjects_.resize(N_OBJECTS);
        for (uint32_t i = 0; i < N_OBJECTS; ++i) {
            sceneObjects_[i].boundingSphere = {
                distXZ(rng), distY(rng), distXZ(rng), distR(rng)
            };
            sceneObjects_[i].vertexCount   = 3;         // 每个物体 = 1 个三角形
            sceneObjects_[i].firstVertex   = i * 3;
        }
    }

    void createBuffers()
    {
        // ── 物体描述缓冲（传给 Compute Shader） ─────────────────────────────
        VkDeviceSize objSz = sizeof(DrawObject) * N_OBJECTS;
        createBuffer(objSz,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, objectBuffer_, objectMemory_);

        // 通过 Staging 上传物体数据
        VkBuffer sb; VkDeviceMemory sm;
        createBuffer(objSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sb, sm);
        void* d = nullptr; vkMapMemory(device_, sm, 0, objSz, 0, &d);
        std::memcpy(d, sceneObjects_.data(), (size_t)objSz); vkUnmapMemory(device_, sm);
        copyBuffer(sb, objectBuffer_, objSz);
        vkDestroyBuffer(device_, sb, nullptr); vkFreeMemory(device_, sm, nullptr);

        // ── 间接绘制命令缓冲（Compute Shader 填写） ──────────────────────────
        VkDeviceSize indSz = sizeof(VkDrawIndirectCommand_) * N_OBJECTS;
        createBuffer(indSz,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |    // Compute 写入
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,    // vkCmdDrawIndirect 读取
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indirectBuffer_, indirectMemory_);

        // ── 可见计数缓冲 ─────────────────────────────────────────────────────
        createBuffer(sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, countBuffer_, countMemory_);

        // ── 顶点缓冲（简单三角形，所有物体共享） ─────────────────────────────
        // 真实项目中每个物体有自己的网格，通过 firstVertex 索引
        std::vector<SceneVertex> verts;
        std::mt19937 rng2(123);
        std::uniform_real_distribution<float> col(0.3f, 1.0f);
        for (uint32_t i = 0; i < N_OBJECTS; ++i) {
            float cx = sceneObjects_[i].boundingSphere.x;
            float cy = sceneObjects_[i].boundingSphere.y;
            float cz = sceneObjects_[i].boundingSphere.z;
            float r  = sceneObjects_[i].boundingSphere.w;
            glm::vec3 c = {col(rng2), col(rng2), col(rng2)};
            verts.push_back({{cx,      cy+r, cz}, c});
            verts.push_back({{cx+r,    cy-r, cz}, c});
            verts.push_back({{cx-r,    cy-r, cz}, c});
        }
        VkDeviceSize vSz = sizeof(SceneVertex) * verts.size();
        createBuffer(vSz,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer_, vertexMemory_);
        VkBuffer vsb; VkDeviceMemory vsm;
        createBuffer(vSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vsb, vsm);
        void* vd = nullptr; vkMapMemory(device_, vsm, 0, vSz, 0, &vd);
        std::memcpy(vd, verts.data(), (size_t)vSz); vkUnmapMemory(device_, vsm);
        copyBuffer(vsb, vertexBuffer_, vSz);
        vkDestroyBuffer(device_, vsb, nullptr); vkFreeMemory(device_, vsm, nullptr);

        std::cout << "✅ 缓冲已创建：物体(" << objSz/1024 << "KB) 间接("
                  << indSz/1024 << "KB) 顶点(" << vSz/1024 << "KB)\n";
    }

    void createCullDescriptorSetLayout()
    {
        std::array<VkDescriptorSetLayoutBinding,3> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        bindings[1]={1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        bindings[2]={2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;ci.bindingCount=static_cast<uint32_t>(bindings.size());ci.pBindings=bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_,&ci,nullptr,&cullSetLayout_));
    }

    void createCullDescriptorSet()
    {
        std::array<VkDescriptorPoolSize,1> ps{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}}};
        VkDescriptorPoolCreateInfo poolCI{};poolCI.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;poolCI.poolSizeCount=1;poolCI.pPoolSizes=ps.data();poolCI.maxSets=1;
        VK_CHECK(vkCreateDescriptorPool(device_,&poolCI,nullptr,&cullDescPool_));
        VkDescriptorSetAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;ai.descriptorPool=cullDescPool_;ai.descriptorSetCount=1;ai.pSetLayouts=&cullSetLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_,&ai,&cullDescSet_));
        VkDescriptorBufferInfo objBI{objectBuffer_,0,VK_WHOLE_SIZE};
        VkDescriptorBufferInfo indBI{indirectBuffer_,0,VK_WHOLE_SIZE};
        VkDescriptorBufferInfo cntBI{countBuffer_,0,VK_WHOLE_SIZE};
        std::array<VkWriteDescriptorSet,3> ws{};
        ws[0]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,cullDescSet_,0,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&objBI,nullptr};
        ws[1]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,cullDescSet_,1,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&indBI,nullptr};
        ws[2]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,cullDescSet_,2,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&cntBI,nullptr};
        vkUpdateDescriptorSets(device_,3,ws.data(),0,nullptr);
    }

    void createCullPipeline()
    {
        VkShaderModule comp = createShaderModuleFromFile(device_, "gpu_cull.comp.spv");
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;stage.module=comp;stage.pName="main";
        VkPushConstantRange pcRange{};pcRange.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;pcRange.offset=0;pcRange.size=sizeof(CullPushConstants);
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.setLayoutCount=1;pli.pSetLayouts=&cullSetLayout_;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_,&pli,nullptr,&cullPipeLayout_));
        VkComputePipelineCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;ci.stage=stage;ci.layout=cullPipeLayout_;
        VK_CHECK(vkCreateComputePipelines(device_,VK_NULL_HANDLE,1,&ci,nullptr,&cullPipeline_));
        vkDestroyShaderModule(device_,comp,nullptr);
        std::cout<<"✅ GPU 剔除计算管线创建完成\n";
    }

    void createScenePipeline()
    {
        VkShaderModule vert = createShaderModuleFromFile(device_, "uniform3d.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "triangle.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vert,"main",nullptr};
        stages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,frag,"main",nullptr};
        VkVertexInputBindingDescription bind{0,sizeof(SceneVertex),VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription,2> attrs{};
        attrs[0]={0,0,VK_FORMAT_R32G32B32_SFLOAT,0};
        attrs[1]={1,0,VK_FORMAT_R32G32B32_SFLOAT,sizeof(glm::vec3)};
        VkPipelineVertexInputStateCreateInfo vi{};vi.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;vi.vertexBindingDescriptionCount=1;vi.pVertexBindingDescriptions=&bind;vi.vertexAttributeDescriptionCount=2;vi.pVertexAttributeDescriptions=attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,nullptr,0,VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,nullptr,0,1,nullptr,1,nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};rs.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;rs.polygonMode=VK_POLYGON_MODE_FILL;rs.lineWidth=1.0f;rs.cullMode=VK_CULL_MODE_NONE;rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};ms.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};cba.colorWriteMask=0xf;
        VkPipelineColorBlendStateCreateInfo cb{};cb.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;cb.attachmentCount=1;cb.pAttachments=&cba;
        std::vector<VkDynamicState> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};dynS.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;dynS.dynamicStateCount=static_cast<uint32_t>(dyn.size());dynS.pDynamicStates=dyn.data();
        // UBO 被 push constant 替代
        VkPushConstantRange pcRange{};pcRange.stageFlags=VK_SHADER_STAGE_VERTEX_BIT;pcRange.offset=0;pcRange.size=sizeof(ScenePushConstants);
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_,&pli,nullptr,&scenePipeLayout_));
        VkGraphicsPipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;pi.stageCount=2;pi.pStages=stages;pi.pVertexInputState=&vi;pi.pInputAssemblyState=&ia;pi.pViewportState=&vs;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pColorBlendState=&cb;pi.pDynamicState=&dynS;pi.layout=scenePipeLayout_;pi.renderPass=renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&scenePipeline_));
        vkDestroyShaderModule(device_,frag,nullptr);vkDestroyShaderModule(device_,vert,nullptr);
        std::cout<<"✅ 场景图形管线创建完成（支持 vkCmdDrawIndirect）\n";
    }

    // 从 MVP 矩阵提取视锥体的 6 个平面
    static std::array<glm::vec4,6> extractFrustumPlanes(const glm::mat4& vp)
    {
        std::array<glm::vec4,6> planes;
        // Gribb-Hartmann 方法：从 VP 矩阵的行提取平面
        for(int i=0;i<4;i++) planes[0][i]=vp[i][3]+vp[i][0];  // Left
        for(int i=0;i<4;i++) planes[1][i]=vp[i][3]-vp[i][0];  // Right
        for(int i=0;i<4;i++) planes[2][i]=vp[i][3]+vp[i][1];  // Bottom
        for(int i=0;i<4;i++) planes[3][i]=vp[i][3]-vp[i][1];  // Top
        for(int i=0;i<4;i++) planes[4][i]=vp[i][3]+vp[i][2];  // Near
        for(int i=0;i<4;i++) planes[5][i]=vp[i][3]-vp[i][2];  // Far
        for(auto&p:planes){float len=glm::length(glm::vec3(p));p/=len;}
        return planes;
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
    {
        static auto start=std::chrono::high_resolution_clock::now();
        float t=std::chrono::duration<float>(std::chrono::high_resolution_clock::now()-start).count();

        VkCommandBufferBeginInfo bi{};bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd,&bi));

        // ── 计算相机矩阵 ──────────────────────────────────────────────────
        float aspect=(float)swapchainExtent_.width/swapchainExtent_.height;
        glm::mat4 proj=glm::perspective(glm::radians(60.0f),aspect,0.5f,50.0f);
        proj[1][1]*=-1;
        glm::vec3 camPos(20*sin(t*0.2f),8,20*cos(t*0.2f));
        glm::mat4 view=glm::lookAt(camPos,glm::vec3(0),glm::vec3(0,1,0));
        glm::mat4 vp=proj*view;

        // ════════════════════════════════════════════════════════════════════
        // 阶段 1: Compute Pass — GPU 视锥体剔除
        // ════════════════════════════════════════════════════════════════════
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,cullPipeline_);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,
            cullPipeLayout_,0,1,&cullDescSet_,0,nullptr);

        CullPushConstants cullPC{};
        cullPC.viewProjection=vp;
        auto planes=extractFrustumPlanes(vp);
        for(int i=0;i<6;++i)cullPC.frustumPlanes[i]=planes[i];
        cullPC.objectCount=N_OBJECTS;
        vkCmdPushConstants(cmd,cullPipeLayout_,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cullPC),&cullPC);

        // 派发剔除：每 64 个物体一个 workgroup
        vkCmdDispatch(cmd,(N_OBJECTS+63)/64,1,1);

        // ── Pipeline Barrier：等待 Compute 写完 Indirect Buffer ──────────
        // 确保间接绘制缓冲在被 vkCmdDrawIndirect 读取前已经填写完毕
        VkBufferMemoryBarrier bufBarrier{};
        bufBarrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bufBarrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;     // Compute 写完
        bufBarrier.dstAccessMask       = VK_ACCESS_INDIRECT_COMMAND_READ_BIT; // 间接读取
        bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufBarrier.buffer              = indirectBuffer_;
        bufBarrier.offset              = 0; bufBarrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,   // 间接绘制读取阶段
            0, 0, nullptr, 1, &bufBarrier, 0, nullptr);

        // ════════════════════════════════════════════════════════════════════
        // 阶段 2: Graphics Pass — 间接绘制
        // ════════════════════════════════════════════════════════════════════
        VkClearValue clear{};clear.color.float32[0]=0.02f;clear.color.float32[2]=0.05f;clear.color.float32[3]=1.0f;
        VkRenderPassBeginInfo rp{};rp.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;rp.renderPass=renderPass_;rp.framebuffer=framebuffers_[imageIndex];rp.renderArea={{0,0},swapchainExtent_};rp.clearValueCount=1;rp.pClearValues=&clear;
        vkCmdBeginRenderPass(cmd,&rp,VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,scenePipeline_);
        VkViewport vp2{0,0,(float)swapchainExtent_.width,(float)swapchainExtent_.height,0,1};vkCmdSetViewport(cmd,0,1,&vp2);
        VkRect2D sc{{0,0},swapchainExtent_};vkCmdSetScissor(cmd,0,1,&sc);
        VkBuffer vb[]={vertexBuffer_};VkDeviceSize off[]={0};
        vkCmdBindVertexBuffers(cmd,0,1,vb,off);

        ScenePushConstants scenePC{};scenePC.viewProjection=vp;scenePC.time=t;
        vkCmdPushConstants(cmd,scenePipeLayout_,VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(scenePC),&scenePC);

        // ── 核心：vkCmdDrawIndirect ──────────────────────────────────────────
        // GPU 直接从 indirectBuffer_ 读取绘制参数
        // instanceCount=0 的条目会被 GPU 自动跳过（被剔除的物体）
        vkCmdDrawIndirect(cmd,
            indirectBuffer_,                          // 间接命令缓冲
            0,                                        // 偏移
            N_OBJECTS,                                // 命令数量
            sizeof(VkDrawIndirectCommand_));          // 每条命令的步长

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
        VkSwapchainKHR scs[]={swapchain_};
        VkPresentInfoKHR pi{};pi.sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;pi.waitSemaphoreCount=1;pi.pWaitSemaphores=ss;pi.swapchainCount=1;pi.pSwapchains=scs;pi.pImageIndices=&imgIdx;
        r=vkQueuePresentKHR(presentQueue_,&pi);
        if(r==VK_ERROR_OUT_OF_DATE_KHR||r==VK_SUBOPTIMAL_KHR||resized_){resized_=false;recreateSwapchain();}
        currentFrame_=(currentFrame_+1)%MAX_FRAMES;
    }

    void mainLoop()
    {
        std::cout<<"🌍 " << N_OBJECTS << " 个物体（旋转相机时观察 GPU 剔除效果）...\n";
        uint64_t frames=0;auto timer=std::chrono::steady_clock::now();
        while(!glfwWindowShouldClose(window_)){
            glfwPollEvents();
            if(glfwGetKey(window_,GLFW_KEY_ESCAPE)==GLFW_PRESS)glfwSetWindowShouldClose(window_,GLFW_TRUE);
            drawFrame();++frames;
            auto now=std::chrono::steady_clock::now();
            float elapsed=std::chrono::duration<float>(now-timer).count();
            if(elapsed>=3.0f){std::cout<<"📊 FPS: "<<(int)(frames/elapsed)<<" | 物体: "<<N_OBJECTS<<" (GPU剔除后可见数取决于视野)\n";frames=0;timer=now;}
        }
        vkDeviceWaitIdle(device_);
    }

    // ─── 辅助函数 ─────────────────────────────────────────────────────────────

    uint32_t findMemoryType(uint32_t f,VkMemoryPropertyFlags p){VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(physicalDevice_,&mp);for(uint32_t i=0;i<mp.memoryTypeCount;++i)if((f&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&p)==p)return i;throw std::runtime_error("找不到内存类型");}
    void createBuffer(VkDeviceSize sz,VkBufferUsageFlags u,VkMemoryPropertyFlags p,VkBuffer&b,VkDeviceMemory&m){VkBufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;ci.size=sz;ci.usage=u;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;VK_CHECK(vkCreateBuffer(device_,&ci,nullptr,&b));VkMemoryRequirements mr;vkGetBufferMemoryRequirements(device_,b,&mr);VkMemoryAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;ai.allocationSize=mr.size;ai.memoryTypeIndex=findMemoryType(mr.memoryTypeBits,p);VK_CHECK(vkAllocateMemory(device_,&ai,nullptr,&m));VK_CHECK(vkBindBufferMemory(device_,b,m,0));}
    void copyBuffer(VkBuffer s,VkBuffer d,VkDeviceSize sz){VkCommandBufferAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandPool=commandPool_;ai.commandBufferCount=1;VkCommandBuffer cmd=VK_NULL_HANDLE;vkAllocateCommandBuffers(device_,&ai,&cmd);VkCommandBufferBeginInfo bi{};bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;vkBeginCommandBuffer(cmd,&bi);VkBufferCopy r{};r.size=sz;vkCmdCopyBuffer(cmd,s,d,1,&r);vkEndCommandBuffer(cmd);VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;si.commandBufferCount=1;si.pCommandBuffers=&cmd;vkQueueSubmit(graphicsQueue_,1,&si,VK_NULL_HANDLE);vkQueueWaitIdle(graphicsQueue_);vkFreeCommandBuffers(device_,commandPool_,1,&cmd);}
    void createInstance(){VkApplicationInfo ai{};ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;ai.apiVersion=VK_API_VERSION_1_3;auto exts=getRequiredInstanceExtensions();VkInstanceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ci.pApplicationInfo=&ai;ci.enabledExtensionCount=static_cast<uint32_t>(exts.size());ci.ppEnabledExtensionNames=exts.data();ci.flags|=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}VK_CHECK(vkCreateInstance(&ci,nullptr,&instance_));}
    void createSurface(){VK_CHECK(glfwCreateWindowSurface(instance_,window_,nullptr,&surface_));}
    void pickPhysicalDevice(){uint32_t c=0;vkEnumeratePhysicalDevices(instance_,&c,nullptr);std::vector<VkPhysicalDevice> devs(c);vkEnumeratePhysicalDevices(instance_,&c,devs.data());for(auto&d:devs)if(findQueueFamilies(d,surface_).isComplete()&&checkDeviceExtensionSupport(d)){physicalDevice_=d;break;}if(!physicalDevice_)throw std::runtime_error("无合适GPU");VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(physicalDevice_,&p);std::cout<<"✅ GPU: "<<p.deviceName<<"\n";}
    void createLogicalDevice(){queueIndices_=findQueueFamilies(physicalDevice_,surface_);std::set<uint32_t> fams={queueIndices_.graphicsFamily.value(),queueIndices_.presentFamily.value()};const float pri=1.0f;std::vector<VkDeviceQueueCreateInfo> qcis;for(uint32_t f:fams){VkDeviceQueueCreateInfo q{};q.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;q.queueFamilyIndex=f;q.queueCount=1;q.pQueuePriorities=&pri;qcis.push_back(q);}VkPhysicalDeviceFeatures feat{};feat.samplerAnisotropy=VK_TRUE;VkDeviceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;ci.queueCreateInfoCount=static_cast<uint32_t>(qcis.size());ci.pQueueCreateInfos=qcis.data();ci.pEnabledFeatures=&feat;ci.enabledExtensionCount=static_cast<uint32_t>(DEVICE_EXTENSIONS.size());ci.ppEnabledExtensionNames=DEVICE_EXTENSIONS.data();if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}VK_CHECK(vkCreateDevice(physicalDevice_,&ci,nullptr,&device_));vkGetDeviceQueue(device_,queueIndices_.graphicsFamily.value(),0,&graphicsQueue_);vkGetDeviceQueue(device_,queueIndices_.presentFamily.value(),0,&presentQueue_);computeQueue_=graphicsQueue_;}
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
        vkDestroyBuffer(device_,objectBuffer_,nullptr);vkFreeMemory(device_,objectMemory_,nullptr);
        vkDestroyBuffer(device_,indirectBuffer_,nullptr);vkFreeMemory(device_,indirectMemory_,nullptr);
        vkDestroyBuffer(device_,countBuffer_,nullptr);vkFreeMemory(device_,countMemory_,nullptr);
        vkDestroyBuffer(device_,vertexBuffer_,nullptr);vkFreeMemory(device_,vertexMemory_,nullptr);
        vkDestroyDescriptorPool(device_,cullDescPool_,nullptr);
        vkDestroyDescriptorSetLayout(device_,cullSetLayout_,nullptr);
        for(int i=0;i<MAX_FRAMES;++i){vkDestroySemaphore(device_,imageAvailableSems_[i],nullptr);vkDestroySemaphore(device_,renderFinishedSems_[i],nullptr);vkDestroyFence(device_,inFlightFences_[i],nullptr);}
        vkDestroyCommandPool(device_,commandPool_,nullptr);for(auto&fb:framebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);
        vkDestroyPipeline(device_,cullPipeline_,nullptr);vkDestroyPipelineLayout(device_,cullPipeLayout_,nullptr);
        vkDestroyPipeline(device_,scenePipeline_,nullptr);vkDestroyPipelineLayout(device_,scenePipeLayout_,nullptr);
        vkDestroyRenderPass(device_,renderPass_,nullptr);for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);
        vkDestroySwapchainKHR(device_,swapchain_,nullptr);vkDestroyDevice(device_,nullptr);vkDestroySurfaceKHR(instance_,surface_,nullptr);vkDestroyInstance(instance_,nullptr);glfwDestroyWindow(window_);glfwTerminate();
        std::cout<<"✅ 清理完成。\n";
    }
};

int main()
{
    std::cout<<"═══════════════════════════════════════════════════════════════════\n";
    std::cout<<" 第23章：GPU-Driven Rendering（间接绘制 + GPU 视锥体剔除）\n";
    std::cout<<"\n";
    std::cout<<" 架构：\n";
    std::cout<<"   Compute Pass: gpu_cull.comp → 填写 VkDrawIndirectCommand\n";
    std::cout<<"   Graphics Pass: vkCmdDrawIndirect → 一次命令绘制所有可见物体\n";
    std::cout<<"\n";
    std::cout<<" 关键 API：\n";
    std::cout<<"   vkCmdDrawIndirect(indirectBuffer, offset, N, stride)\n";
    std::cout<<"   instanceCount=0 → GPU 跳过该条命令（被剔除）\n";
    std::cout<<"   Pipeline Barrier → Compute写完再Indirect读\n";
    std::cout<<"═══════════════════════════════════════════════════════════════════\n\n";
    Ch23App app;
    try{app.run();}catch(const std::exception&e){std::cerr<<"❌ "<<e.what()<<"\n";return 1;}
    return 0;
}
