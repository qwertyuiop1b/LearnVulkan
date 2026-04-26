/**
 * @file ch31_shader_object.cpp
 * @brief 第31章：Shader Object（VK_EXT_shader_object）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【什么是 Shader Object？】
 *
 *  Vulkan 1.3 之前，所有渲染状态都必须提前编译到 VkPipeline 中。
 *  这导致：
 *  - 应用启动时要编译大量管线（几秒到几分钟的加载时间）
 *  - 运行时状态改变（换着色器/混合模式）需要切换整个管线
 *  - 管线状态"组合爆炸"：N 种着色器 × M 种状态 = N×M 个管线
 *
 *  VK_EXT_shader_object（2023年，Vulkan 1.3 扩展）完全解决这些问题：
 *    - 着色器单独编译，不绑定到管线
 *    - 所有渲染状态（混合/光栅化/深度等）通过动态命令设置
 *    - 无需 VkPipeline 对象！
 *
 * 【Shader Object vs 传统管线对比】
 *
 *  传统管线：
 *    vkCreateGraphicsPipelines(...)  ← 耗时，一次性
 *    vkCmdBindPipeline(...)          ← 每次切换材质都需要
 *
 *  Shader Object：
 *    vkCreateShadersEXT(...)         ← 只编译着色器，快！
 *    vkCmdBindShadersEXT(...)        ← 绑定着色器
 *    vkCmdSetXxx(...)               ← 动态设置所有状态
 *
 * 【扩展的完整动态状态集】
 *
 *  所有原来在管线中固化的状态，现在都可以动态设置：
 *  vkCmdSetRasterizerDiscardEnableEXT()
 *  vkCmdSetPolygonModeEXT()
 *  vkCmdSetCullModeEXT()
 *  vkCmdSetFrontFaceEXT()
 *  vkCmdSetDepthTestEnableEXT()
 *  vkCmdSetColorBlendEnableEXT()
 *  vkCmdSetColorBlendEquationEXT()
 *  vkCmdSetVertexInputEXT()
 *  ... 等 50+ 个动态状态
 *
 * 【VkShaderEXT 对象】
 *
 *  VkShaderCreateInfoEXT：
 *    - stage         : 着色器阶段（VERTEX / FRAGMENT / etc）
 *    - codeType      : BINARY（SPIR-V）或 SOURCE（GLSL，驱动实时编译）
 *    - pCode/codeSize: SPIR-V 字节码
 *    - pSetLayouts   : 描述符集布局（与管线中相同）
 *
 *  vkCreateShadersEXT(device, count, pCreateInfos, pAllocator, pShaders)
 *    → 可以一次批量创建多个着色器（减少往返次数）
 *
 * 【Linked Shaders 链接着色器】
 *
 *  如果着色器在 pCreateInfos 中相邻，并都设置 LINK_STAGE 标志，
 *  驱动可以一次性链接它们（类似传统管线，但更灵活）。
 *
 * 【本章示例】
 *
 *  - 运行时动态切换着色器（按 1/2/3 键）
 *  - 按 W 键切换线框/实体模式（无需重建管线！）
 *  - 展示 50+ 动态状态 API
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <array>
#include <chrono>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int      MAX_FRAMES = 2;

struct SOVertex { float pos[2]; float col[3]; };

static const std::vector<SOVertex> TRIANGLE = {
    {{ 0.0f,-0.6f},{1,0.2f,0.2f}},
    {{ 0.6f, 0.5f},{0.2f,1,0.2f}},
    {{-0.6f, 0.5f},{0.2f,0.2f,1}},
};

class Ch31App {
public:
    void run()
    {
        initWindow();
        if (!initVulkan()) { printGuide(); glfwDestroyWindow(window_); glfwTerminate(); return; }
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
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;

    // ─── Shader Object（替代 VkPipeline！）────────────────────────────────
    VkShaderEXT  vertShader_[3] = {};   // 3 种顶点着色器（动态切换）
    VkShaderEXT  fragShader_[2] = {};   // 2 种片段着色器
    int          activeVert_    = 0;
    int          activeFrag_    = 0;
    bool         wireframe_     = false;

    // Shader Object 函数指针
    PFN_vkCreateShadersEXT          fpCreateShaders_          = nullptr;
    PFN_vkDestroyShaderEXT          fpDestroyShader_          = nullptr;
    PFN_vkCmdBindShadersEXT         fpCmdBindShaders_         = nullptr;
    PFN_vkCmdSetVertexInputEXT      fpCmdSetVertexInput_      = nullptr;
    PFN_vkCmdSetPolygonModeEXT      fpCmdSetPolygonMode_      = nullptr;
    PFN_vkCmdSetCullModeEXT         fpCmdSetCullMode_         = nullptr;
    PFN_vkCmdSetFrontFaceEXT        fpCmdSetFrontFace_        = nullptr;
    PFN_vkCmdSetColorBlendEnableEXT fpCmdSetColorBlendEnable_ = nullptr;

    VkBuffer       vertexBuffer_  = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_  = VK_NULL_HANDLE;

    std::vector<VkImage>         swapchainImages_;
    std::vector<VkImageView>     swapchainImageViews_;
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
            "Ch31 - Shader Object（无管线对象！）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_,
            [](GLFWwindow*w,int,int){
                reinterpret_cast<Ch31App*>(glfwGetWindowUserPointer(w))->resized_=true;
            });
        glfwSetKeyCallback(window_, [](GLFWwindow*w,int k,int,int a,int) {
            if(a!=GLFW_PRESS)return;
            auto* app=reinterpret_cast<Ch31App*>(glfwGetWindowUserPointer(w));
            if(k==GLFW_KEY_1)app->activeVert_=0;
            else if(k==GLFW_KEY_2)app->activeVert_=1;
            else if(k==GLFW_KEY_3)app->activeVert_=2;
            else if(k==GLFW_KEY_W)app->wireframe_=!app->wireframe_;
            std::cout<<"Shader: vert["<<app->activeVert_<<"]  Wireframe="<<app->wireframe_<<"\n";
        });
    }

    bool initVulkan()
    {
        try {
            createInstance(); createSurface();
            if (!pickPhysicalDeviceSO()) return false;
            createLogicalDeviceSO();
            loadSOFunctionPointers();
            createSwapchain(); createImageViews();
            createShaderObjects();   // ← 替代 vkCreateGraphicsPipelines！
            createCommandPool();
            createVertexBuffer();
            createCommandBuffers(); createSyncObjects();
            std::cout << "\n✅ Shader Object 初始化完成（无 VkPipeline 对象）！\n";
            std::cout << "🔑 1/2/3：切换着色器  W：线框/实体  ESC：退出\n";
        } catch (const std::exception& e) {
            std::cerr << "⚠️  " << e.what() << "\n"; return false;
        }
        return true;
    }

    bool pickPhysicalDeviceSO()
    {
        uint32_t c=0; vkEnumeratePhysicalDevices(instance_,&c,nullptr);
        std::vector<VkPhysicalDevice> devs(c);
        vkEnumeratePhysicalDevices(instance_,&c,devs.data());
        for (auto& d : devs) {
            if (!findQueueFamilies(d,surface_).isComplete()) continue;
            uint32_t ec=0; vkEnumerateDeviceExtensionProperties(d,nullptr,&ec,nullptr);
            std::vector<VkExtensionProperties> exts(ec);
            vkEnumerateDeviceExtensionProperties(d,nullptr,&ec,exts.data());
            bool hasSOExt = false;
            for (auto& e : exts)
                if (std::string(e.extensionName) == "VK_EXT_shader_object") { hasSOExt=true; break; }
            if (!hasSOExt) { std::cout<<"⚠️  不支持 VK_EXT_shader_object\n"; continue; }

            // 检查特性
            VkPhysicalDeviceShaderObjectFeaturesEXT soFeatures{};
            soFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
            VkPhysicalDeviceFeatures2 feat2{};
            feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            feat2.pNext = &soFeatures;
            vkGetPhysicalDeviceFeatures2(d,&feat2);
            if (!soFeatures.shaderObject) continue;

            physicalDevice_=d;
            VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d,&p);
            std::cout<<"✅ GPU: "<<p.deviceName<<" (Shader Object 支持)\n";
            return true;
        }
        return false;
    }

    void createLogicalDeviceSO()
    {
        queueIndices_=findQueueFamilies(physicalDevice_,surface_);
        std::set<uint32_t> fams={queueIndices_.graphicsFamily.value(),queueIndices_.presentFamily.value()};
        const float pri=1.0f; std::vector<VkDeviceQueueCreateInfo> qcis;
        for(uint32_t f:fams){ VkDeviceQueueCreateInfo q{};q.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;q.queueFamilyIndex=f;q.queueCount=1;q.pQueuePriorities=&pri;qcis.push_back(q); }

        // 启用 Shader Object 特性
        VkPhysicalDeviceShaderObjectFeaturesEXT soFeatures{};
        soFeatures.sType        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
        soFeatures.shaderObject = VK_TRUE;

        // 还需要启用 Dynamic Rendering（Shader Object 依赖）
        VkPhysicalDeviceDynamicRenderingFeatures dynRender{};
        dynRender.sType           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
        dynRender.dynamicRendering = VK_TRUE;
        dynRender.pNext           = &soFeatures;

        static const std::vector<const char*> soExts = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            "VK_KHR_portability_subset",
            "VK_EXT_shader_object",
            // Shader Object 需要的依赖扩展：
            "VK_KHR_dynamic_rendering",
            "VK_EXT_extended_dynamic_state",
            "VK_EXT_extended_dynamic_state2",
            "VK_EXT_extended_dynamic_state3",
            "VK_EXT_vertex_input_dynamic_state",
        };

        VkPhysicalDeviceFeatures feat{}; feat.fillModeNonSolid = VK_TRUE;
        VkDeviceCreateInfo ci{};
        ci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;ci.pNext=&dynRender;
        ci.queueCreateInfoCount=static_cast<uint32_t>(qcis.size());ci.pQueueCreateInfos=qcis.data();
        ci.pEnabledFeatures=&feat;
        ci.enabledExtensionCount=static_cast<uint32_t>(soExts.size());
        ci.ppEnabledExtensionNames=soExts.data();
        if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}
        VK_CHECK(vkCreateDevice(physicalDevice_,&ci,nullptr,&device_));
        vkGetDeviceQueue(device_,queueIndices_.graphicsFamily.value(),0,&graphicsQueue_);
        vkGetDeviceQueue(device_,queueIndices_.presentFamily.value(),0,&presentQueue_);
        std::cout<<"✅ 逻辑设备已创建（Shader Object + Dynamic Rendering）\n";
    }

    void loadSOFunctionPointers()
    {
        auto load=[this](const char* n,void** p){
            *p=reinterpret_cast<void*>(vkGetDeviceProcAddr(device_,n));
            if(!*p)throw std::runtime_error(std::string("无法加载 ")+n);
        };
        load("vkCreateShadersEXT",          reinterpret_cast<void**>(&fpCreateShaders_));
        load("vkDestroyShaderEXT",          reinterpret_cast<void**>(&fpDestroyShader_));
        load("vkCmdBindShadersEXT",         reinterpret_cast<void**>(&fpCmdBindShaders_));
        load("vkCmdSetVertexInputEXT",      reinterpret_cast<void**>(&fpCmdSetVertexInput_));
        load("vkCmdSetPolygonModeEXT",      reinterpret_cast<void**>(&fpCmdSetPolygonMode_));
        load("vkCmdSetCullModeEXT",         reinterpret_cast<void**>(&fpCmdSetCullMode_));
        load("vkCmdSetFrontFaceEXT",        reinterpret_cast<void**>(&fpCmdSetFrontFace_));
        load("vkCmdSetColorBlendEnableEXT", reinterpret_cast<void**>(&fpCmdSetColorBlendEnable_));
        std::cout<<"✅ Shader Object 函数指针加载完成\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心：vkCreateShadersEXT（替代 vkCreateGraphicsPipelines）
    // ═══════════════════════════════════════════════════════════════════════

    void createShaderObjects()
    {
        // 加载着色器代码
        auto vertCode = readFile(std::string(SHADER_DIR) + "/uniform3d.vert.spv");
        auto fragCode = readFile(std::string(SHADER_DIR) + "/triangle.frag.spv");

        // ── VkShaderCreateInfoEXT：描述单个着色器（非整个管线！） ──────────
        VkShaderCreateInfoEXT vertCI{};
        vertCI.sType     = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
        vertCI.stage     = VK_SHADER_STAGE_VERTEX_BIT;
        // nextStage：此着色器的下一个阶段（用于链接优化）
        vertCI.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
        // LINK_STAGE：与下一个着色器一起链接（驱动优化）
        vertCI.flags     = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT;
        vertCI.codeType  = VK_SHADER_CODE_TYPE_SPIRV_EXT;
        vertCI.codeSize  = vertCode.size();
        vertCI.pCode     = reinterpret_cast<const uint32_t*>(vertCode.data());
        vertCI.pName     = "main";

        VkShaderCreateInfoEXT fragCI{};
        fragCI.sType     = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
        fragCI.stage     = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragCI.nextStage = 0;  // 片段着色器后没有更多阶段
        fragCI.flags     = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT;
        fragCI.codeType  = VK_SHADER_CODE_TYPE_SPIRV_EXT;
        fragCI.codeSize  = fragCode.size();
        fragCI.pCode     = reinterpret_cast<const uint32_t*>(fragCode.data());
        fragCI.pName     = "main";

        // 同时创建两个着色器（批量创建 = 驱动可以链接优化）
        VkShaderCreateInfoEXT infos[] = {vertCI, fragCI};
        VkShaderEXT          shaders[2] = {};
        VK_CHECK(fpCreateShaders_(device_, 2, infos, nullptr, shaders));
        vertShader_[0] = shaders[0];   // 着色器组合1
        fragShader_[0] = shaders[1];
        // 简化：所有组合使用相同着色器，演示切换 API
        vertShader_[1] = shaders[0];
        vertShader_[2] = shaders[0];
        fragShader_[1] = shaders[1];

        std::cout << "✅ Shader Object 已创建（无 VkPipeline！）\n";
        std::cout << "   vkCreateShadersEXT 创建了 2 个 VkShaderEXT 对象\n";
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo bi{};bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd,&bi));

        // 布局转换
        VkImageMemoryBarrier barrier{};
        barrier.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED;barrier.newLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.image=swapchainImages_[imageIndex];barrier.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        barrier.srcAccessMask=0;barrier.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,0,0,nullptr,0,nullptr,1,&barrier);

        VkClearValue clear{};clear.color.float32[0]=0.02f;clear.color.float32[2]=0.05f;clear.color.float32[3]=1.0f;
        VkRenderingAttachmentInfo att{};
        att.sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        att.imageView=swapchainImageViews_[imageIndex];att.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        att.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;att.storeOp=VK_ATTACHMENT_STORE_OP_STORE;att.clearValue=clear;
        VkRenderingInfo ri{};ri.sType=VK_STRUCTURE_TYPE_RENDERING_INFO;ri.renderArea={{0,0},swapchainExtent_};ri.layerCount=1;ri.colorAttachmentCount=1;ri.pColorAttachments=&att;
        vkCmdBeginRendering(cmd,&ri);

        // ── Shader Object 绑定（替代 vkCmdBindPipeline！）────────────────────
        VkShaderStageFlagBits stages[] = {
            VK_SHADER_STAGE_VERTEX_BIT,
            VK_SHADER_STAGE_FRAGMENT_BIT
        };
        VkShaderEXT shaders[] = {
            vertShader_[activeVert_],
            fragShader_[activeFrag_]
        };
        fpCmdBindShaders_(cmd, 2, stages, shaders);

        // ── 动态设置所有渲染状态（不需要管线！）──────────────────────────────
        VkViewport vp{0,0,(float)swapchainExtent_.width,(float)swapchainExtent_.height,0,1};
        vkCmdSetViewportWithCount(cmd,1,&vp);
        VkRect2D sc{{0,0},swapchainExtent_};vkCmdSetScissorWithCount(cmd,1,&sc);

        // 多边形模式（按 W 键切换）
        fpCmdSetPolygonMode_(cmd, wireframe_ ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL);
        fpCmdSetCullMode_(cmd, VK_CULL_MODE_NONE);
        fpCmdSetFrontFace_(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);

        // 颜色混合（禁用）
        VkBool32 blendEnable = VK_FALSE;
        fpCmdSetColorBlendEnable_(cmd, 0, 1, &blendEnable);

        // 顶点输入格式（动态！无需在管线创建时指定）
        VkVertexInputBindingDescription2EXT binding{};
        binding.sType=VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT;
        binding.binding=0;binding.stride=sizeof(SOVertex);binding.inputRate=VK_VERTEX_INPUT_RATE_VERTEX;binding.divisor=1;

        std::array<VkVertexInputAttributeDescription2EXT,2> attrs{};
        attrs[0].sType=VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT;
        attrs[0].location=0;attrs[0].binding=0;attrs[0].format=VK_FORMAT_R32G32_SFLOAT;attrs[0].offset=0;
        attrs[1].sType=VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT;
        attrs[1].location=1;attrs[1].binding=0;attrs[1].format=VK_FORMAT_R32G32B32_SFLOAT;attrs[1].offset=8;

        fpCmdSetVertexInput_(cmd,1,&binding,2,attrs.data());

        // 深度测试（禁用）
        vkCmdSetDepthTestEnable(cmd, VK_FALSE);
        vkCmdSetDepthWriteEnable(cmd, VK_FALSE);

        // 图元拓扑
        vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);

        // 绑定顶点缓冲 + 绘制
        VkBuffer vb[]={vertexBuffer_};VkDeviceSize off[]={0};
        vkCmdBindVertexBuffers(cmd,0,1,vb,off);
        vkCmdDraw(cmd,3,1,0,0);

        vkCmdEndRendering(cmd);

        // 布局转换 → Present
        barrier.oldLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask=0;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,nullptr,0,nullptr,1,&barrier);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame()
    {
        vkWaitForFences(device_,1,&inFlightFences_[currentFrame_],VK_TRUE,UINT64_MAX);
        uint32_t imgIdx=0;VkResult r=vkAcquireNextImageKHR(device_,swapchain_,UINT64_MAX,imageAvailableSems_[currentFrame_],VK_NULL_HANDLE,&imgIdx);
        if(r==VK_ERROR_OUT_OF_DATE_KHR){recreateSwapchain();return;}
        vkResetFences(device_,1,&inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_],0);
        recordCommandBuffer(commandBuffers_[currentFrame_],imgIdx);
        VkSemaphore ws[]={imageAvailableSems_[currentFrame_]};VkPipelineStageFlags wst[]={VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};VkSemaphore ss[]={renderFinishedSems_[currentFrame_]};
        VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;si.waitSemaphoreCount=1;si.pWaitSemaphores=ws;si.pWaitDstStageMask=wst;si.commandBufferCount=1;si.pCommandBuffers=&commandBuffers_[currentFrame_];si.signalSemaphoreCount=1;si.pSignalSemaphores=ss;
        VK_CHECK(vkQueueSubmit(graphicsQueue_,1,&si,inFlightFences_[currentFrame_]));
        VkSwapchainKHR scs[]={swapchain_};VkPresentInfoKHR pi{};pi.sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;pi.waitSemaphoreCount=1;pi.pWaitSemaphores=ss;pi.swapchainCount=1;pi.pSwapchains=scs;pi.pImageIndices=&imgIdx;
        r=vkQueuePresentKHR(presentQueue_,&pi);
        if(r==VK_ERROR_OUT_OF_DATE_KHR||r==VK_SUBOPTIMAL_KHR||resized_){resized_=false;recreateSwapchain();}
        currentFrame_=(currentFrame_+1)%MAX_FRAMES;
    }

    void mainLoop()
    {
        std::cout<<"🎨 Shader Object 三角形（无管线对象，ESC退出）...\n";
        while(!glfwWindowShouldClose(window_)){glfwPollEvents();if(glfwGetKey(window_,GLFW_KEY_ESCAPE)==GLFW_PRESS)glfwSetWindowShouldClose(window_,GLFW_TRUE);drawFrame();}
        vkDeviceWaitIdle(device_);
    }

    void printGuide()
    {
        std::cout<<"\n Shader Object 概念速查（此 GPU 不支持扩展）\n";
        std::cout<<"  扩展：VK_EXT_shader_object（2023年，NVIDIA/AMD 均支持）\n";
        std::cout<<"  依赖：VK_EXT_extended_dynamic_state{1,2,3}\n";
        std::cout<<"       VK_EXT_vertex_input_dynamic_state\n";
        std::cout<<"       VK_KHR_dynamic_rendering\n\n";
        std::cout<<"  API：\n";
        std::cout<<"   vkCreateShadersEXT(device, count, infos, allocator, shaders)\n";
        std::cout<<"   vkCmdBindShadersEXT(cmd, stageCount, stages, shaders)\n";
        std::cout<<"   + 50+ vkCmdSetXxx 动态状态函数\n\n";
        std::cout<<"  优势：\n";
        std::cout<<"   零管线编译等待  | 运行时随意切换着色器 | 消除状态组合爆炸\n";
    }

    uint32_t findMemoryType(uint32_t f,VkMemoryPropertyFlags p){VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(physicalDevice_,&mp);for(uint32_t i=0;i<mp.memoryTypeCount;++i)if((f&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&p)==p)return i;throw std::runtime_error("找不到内存类型");}
    void createBuffer(VkDeviceSize sz,VkBufferUsageFlags u,VkMemoryPropertyFlags p,VkBuffer&b,VkDeviceMemory&m){VkBufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;ci.size=sz;ci.usage=u;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;VK_CHECK(vkCreateBuffer(device_,&ci,nullptr,&b));VkMemoryRequirements mr;vkGetBufferMemoryRequirements(device_,b,&mr);VkMemoryAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;ai.allocationSize=mr.size;ai.memoryTypeIndex=findMemoryType(mr.memoryTypeBits,p);VK_CHECK(vkAllocateMemory(device_,&ai,nullptr,&m));VK_CHECK(vkBindBufferMemory(device_,b,m,0));}
    void createVertexBuffer(){VkDeviceSize sz=sizeof(TRIANGLE[0])*TRIANGLE.size();createBuffer(sz,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,vertexBuffer_,vertexMemory_);void*d=nullptr;vkMapMemory(device_,vertexMemory_,0,sz,0,&d);std::memcpy(d,TRIANGLE.data(),(size_t)sz);vkUnmapMemory(device_,vertexMemory_);}
    void createInstance(){VkApplicationInfo ai{};ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;ai.apiVersion=VK_API_VERSION_1_3;auto exts=getRequiredInstanceExtensions();VkInstanceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ci.pApplicationInfo=&ai;ci.enabledExtensionCount=static_cast<uint32_t>(exts.size());ci.ppEnabledExtensionNames=exts.data();ci.flags|=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}VK_CHECK(vkCreateInstance(&ci,nullptr,&instance_));}
    void createSurface(){VK_CHECK(glfwCreateWindowSurface(instance_,window_,nullptr,&surface_));}
    void createSwapchain(){auto sc=querySwapChainSupport(physicalDevice_,surface_);auto fmt=chooseSwapSurfaceFormat(sc.formats);auto mode=chooseSwapPresentMode(sc.presentModes);swapchainExtent_=chooseSwapExtent(sc.capabilities,window_);uint32_t n=sc.capabilities.minImageCount+1;if(sc.capabilities.maxImageCount>0)n=std::min(n,sc.capabilities.maxImageCount);VkSwapchainCreateInfoKHR ci{};ci.sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;ci.surface=surface_;ci.minImageCount=n;ci.imageFormat=fmt.format;ci.imageColorSpace=fmt.colorSpace;ci.imageExtent=swapchainExtent_;ci.imageArrayLayers=1;ci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;ci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE;ci.preTransform=sc.capabilities.currentTransform;ci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;ci.presentMode=mode;ci.clipped=VK_TRUE;VK_CHECK(vkCreateSwapchainKHR(device_,&ci,nullptr,&swapchain_));vkGetSwapchainImagesKHR(device_,swapchain_,&n,nullptr);swapchainImages_.resize(n);vkGetSwapchainImagesKHR(device_,swapchain_,&n,swapchainImages_.data());swapchainImageFormat_=fmt.format;}
    void createImageViews(){swapchainImageViews_.resize(swapchainImages_.size());for(size_t i=0;i<swapchainImages_.size();++i){VkImageViewCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;ci.image=swapchainImages_[i];ci.viewType=VK_IMAGE_VIEW_TYPE_2D;ci.format=swapchainImageFormat_;ci.components={VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY};ci.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};VK_CHECK(vkCreateImageView(device_,&ci,nullptr,&swapchainImageViews_[i]));}}
    void createCommandPool(){VkCommandPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;ci.queueFamilyIndex=queueIndices_.graphicsFamily.value();VK_CHECK(vkCreateCommandPool(device_,&ci,nullptr,&commandPool_));}
    void createCommandBuffers(){commandBuffers_.resize(MAX_FRAMES);VkCommandBufferAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;ai.commandPool=commandPool_;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=static_cast<uint32_t>(commandBuffers_.size());VK_CHECK(vkAllocateCommandBuffers(device_,&ai,commandBuffers_.data()));}
    void createSyncObjects(){imageAvailableSems_.resize(MAX_FRAMES);renderFinishedSems_.resize(MAX_FRAMES);inFlightFences_.resize(MAX_FRAMES);VkSemaphoreCreateInfo sCI{};sCI.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;VkFenceCreateInfo fCI{};fCI.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;fCI.flags=VK_FENCE_CREATE_SIGNALED_BIT;for(int i=0;i<MAX_FRAMES;++i){VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&imageAvailableSems_[i]));VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&renderFinishedSems_[i]));VK_CHECK(vkCreateFence(device_,&fCI,nullptr,&inFlightFences_[i]));}}
    void recreateSwapchain(){int w=0,h=0;glfwGetFramebufferSize(window_,&w,&h);while(!w||!h){glfwGetFramebufferSize(window_,&w,&h);glfwWaitEvents();}vkDeviceWaitIdle(device_);for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);vkDestroySwapchainKHR(device_,swapchain_,nullptr);createSwapchain();createImageViews();}
    void cleanup()
    {
        for(int i=0;i<3;++i)if(vertShader_[i]&&fpDestroyShader_)fpDestroyShader_(device_,vertShader_[i],nullptr);
        if(fragShader_[0]&&fpDestroyShader_)fpDestroyShader_(device_,fragShader_[0],nullptr);
        vkDestroyBuffer(device_,vertexBuffer_,nullptr);vkFreeMemory(device_,vertexMemory_,nullptr);
        for(int i=0;i<MAX_FRAMES;++i){vkDestroySemaphore(device_,imageAvailableSems_[i],nullptr);vkDestroySemaphore(device_,renderFinishedSems_[i],nullptr);vkDestroyFence(device_,inFlightFences_[i],nullptr);}
        vkDestroyCommandPool(device_,commandPool_,nullptr);
        for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);
        vkDestroySwapchainKHR(device_,swapchain_,nullptr);vkDestroyDevice(device_,nullptr);
        vkDestroySurfaceKHR(instance_,surface_,nullptr);vkDestroyInstance(instance_,nullptr);
        glfwDestroyWindow(window_);glfwTerminate();std::cout<<"✅ 清理完成。\n";
    }
};

int main()
{
    std::cout<<"═══════════════════════════════════════════════════════════════════\n";
    std::cout<<" 第31章：Shader Object（VK_EXT_shader_object，2023）\n";
    std::cout<<"\n";
    std::cout<<" 核心思路：用 VkShaderEXT 替代 VkPipeline\n";
    std::cout<<"   vkCreateShadersEXT    → 创建独立着色器对象\n";
    std::cout<<"   vkCmdBindShadersEXT   → 绑定着色器（替代 vkCmdBindPipeline）\n";
    std::cout<<"   vkCmdSetPolygonModeEXT → 50+ 动态状态（替代管线固化状态）\n";
    std::cout<<"\n";
    std::cout<<" 优势：零管线编译等待 | 运行时随意换着色器 | 无状态组合爆炸\n";
    std::cout<<"═══════════════════════════════════════════════════════════════════\n\n";
    Ch31App app;
    try{app.run();}catch(const std::exception&e){std::cerr<<"❌ "<<e.what()<<"\n";return 1;}
    return 0;
}
