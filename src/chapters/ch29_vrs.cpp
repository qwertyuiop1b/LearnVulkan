/**
 * @file ch29_vrs.cpp
 * @brief 第29章：Variable Rate Shading（VRS / 可变着色率）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【什么是 VRS？】
 *
 *  传统渲染：每个像素运行一次片段着色器（1x1 采样率）
 *  VRS：不同屏幕区域可以使用不同着色率，以节省 GPU 计算
 *
 *  着色率选项（每"片段"覆盖多少像素）：
 *    1x1: 每像素一次着色（全精度）
 *    1x2: 每 2 个垂直像素共享一次着色
 *    2x1: 每 2 个水平像素共享一次着色
 *    2x2: 每 4 像素（2x2 块）共享一次着色
 *    4x2, 2x4, 4x4: 更低精度（更多像素共享）
 *
 * 【三种 VRS 模式】
 *
 *  1. 管线级别（Pipeline Rate）：
 *     整个绘制命令使用同一着色率
 *     vkCmdSetFragmentShadingRateKHR(rate, combinerOps)
 *
 *  2. 图元级别（Primitive Rate）：
 *     顶点着色器通过 PrimitiveShadingRateNV 输出每个图元的着色率
 *     可以根据物体到相机距离动态选择
 *
 *  3. 附件级别（Attachment Rate / Fragment Density Map）：
 *     创建一张专用的"着色率附件"纹理（VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT）
 *     每个"瓦片"（如 16x16 像素）对应纹理中一个像素，存储着色率
 *     可以实现：
 *       - 中央高精度（VR 注视点渲染）
 *       - 运动模糊区域低精度
 *       - 遮挡区域超低精度
 *
 * 【VR 注视点渲染（Foveated Rendering）】
 *
 *  眼球追踪 → 确定注视点位置 → 注视点附近 1x1，周边 4x4
 *  节省 50-75% 的片段着色器计算，人眼几乎察觉不到差异
 *
 * 【所需扩展】
 *  VK_KHR_fragment_shading_rate
 *
 * 【本章实现】
 *  1. 查询设备 VRS 能力（支持的着色率列表）
 *  2. 用管线级别 VRS 渲染不同区域（可视化着色率边界）
 *  3. 实时展示 FPS 对比（全精度 vs VRS）
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <chrono>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int      MAX_FRAMES = 2;

struct VRSPushConstants {
    float resolution[2];
    float time;
};

class Ch29App {
public:
    void run()
    {
        initWindow();
        if (!initVulkan()) {
            printVRSGuide();
            glfwDestroyWindow(window_);
            glfwTerminate();
            return;
        }
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
    VkPipeline       pipeline_       = VK_NULL_HANDLE;
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;

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

    // VRS 函数指针
    PFN_vkCmdSetFragmentShadingRateKHR fpSetShadingRate_ = nullptr;
    bool vrsSupported_ = false;

    void initWindow()
    {
        glfwInit(); glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT,
            "Ch29 - Variable Rate Shading（可变着色率）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_,
            [](GLFWwindow*w,int,int){
                reinterpret_cast<Ch29App*>(glfwGetWindowUserPointer(w))->resized_=true;
            });
    }

    bool initVulkan()
    {
        try {
            createInstance();
            createSurface();
            if (!pickPhysicalDeviceVRS()) return false;
            createLogicalDeviceVRS();
            fpSetShadingRate_ = reinterpret_cast<PFN_vkCmdSetFragmentShadingRateKHR>(
                vkGetDeviceProcAddr(device_, "vkCmdSetFragmentShadingRateKHR"));
            createSwapchain(); createImageViews(); createRenderPass();
            createGraphicsPipeline();
            createFramebuffers(); createCommandPool();
            createCommandBuffers(); createSyncObjects();
            vrsSupported_ = true;
            std::cout << "\n✅ VRS 初始化成功！\n";
        } catch (const std::exception& e) {
            std::cerr << "⚠️  " << e.what() << "\n";
            return false;
        }
        return true;
    }

    bool pickPhysicalDeviceVRS()
    {
        uint32_t c=0;vkEnumeratePhysicalDevices(instance_,&c,nullptr);
        std::vector<VkPhysicalDevice> devs(c);vkEnumeratePhysicalDevices(instance_,&c,devs.data());

        for (auto& d : devs) {
            if (!findQueueFamilies(d, surface_).isComplete()) continue;

            // 检查 VRS 扩展支持
            uint32_t extCount=0;vkEnumerateDeviceExtensionProperties(d,nullptr,&extCount,nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            vkEnumerateDeviceExtensionProperties(d,nullptr,&extCount,exts.data());

            bool hasVRS = false;
            for (auto& e : exts)
                if (std::string(e.extensionName) == "VK_KHR_fragment_shading_rate")
                    { hasVRS = true; break; }

            if (!hasVRS) { std::cout<<"⚠️  GPU 不支持 VK_KHR_fragment_shading_rate\n"; continue; }

            // 查询支持的着色率
            VkPhysicalDeviceFragmentShadingRateFeaturesKHR vrsFeatures{};
            vrsFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
            VkPhysicalDeviceFeatures2 feat2{};feat2.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;feat2.pNext=&vrsFeatures;
            vkGetPhysicalDeviceFeatures2(d, &feat2);

            if (!vrsFeatures.pipelineFragmentShadingRate) continue;

            physicalDevice_ = d;
            VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(d,&p);
            std::cout<<"✅ GPU: "<<p.deviceName<<"\n";

            // 打印支持的着色率
            // 动态加载 VRS 查询函数（扩展函数必须通过 vkGetInstanceProcAddr 加载）
            auto fpGetRates = reinterpret_cast<PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR>(
                vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceFragmentShadingRatesKHR"));
            if (!fpGetRates) { std::cout<<"⚠️  无法加载 VRS 查询函数\n"; continue; }

            uint32_t rateCount=0;
            fpGetRates(d,&rateCount,nullptr);
            std::vector<VkPhysicalDeviceFragmentShadingRateKHR> rates(rateCount);
            for(auto&r:rates)r.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR;
            fpGetRates(d,&rateCount,rates.data());

            std::cout<<"🔷 支持的着色率（" << rateCount << " 种）：\n";
            for(auto&r:rates)
                std::cout<<"   "<<r.fragmentSize.width<<"x"<<r.fragmentSize.height<<"\n";
            return true;
        }
        return false;
    }

    void createLogicalDeviceVRS()
    {
        queueIndices_=findQueueFamilies(physicalDevice_,surface_);
        std::set<uint32_t> fams={queueIndices_.graphicsFamily.value(),queueIndices_.presentFamily.value()};
        const float pri=1.0f;std::vector<VkDeviceQueueCreateInfo> qcis;
        for(uint32_t f:fams){VkDeviceQueueCreateInfo q{};q.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;q.queueFamilyIndex=f;q.queueCount=1;q.pQueuePriorities=&pri;qcis.push_back(q);}

        VkPhysicalDeviceFragmentShadingRateFeaturesKHR vrsFeatures{};
        vrsFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
        vrsFeatures.pipelineFragmentShadingRate = VK_TRUE;

        static std::vector<const char*> vrsExts = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            "VK_KHR_portability_subset",
            "VK_KHR_fragment_shading_rate",
            "VK_KHR_create_renderpass2",
            "VK_KHR_multiview",
            "VK_KHR_maintenance2",
        };

        VkPhysicalDeviceFeatures feat{};feat.samplerAnisotropy=VK_TRUE;
        VkDeviceCreateInfo ci{};
        ci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;ci.pNext=&vrsFeatures;
        ci.queueCreateInfoCount=static_cast<uint32_t>(qcis.size());ci.pQueueCreateInfos=qcis.data();ci.pEnabledFeatures=&feat;
        ci.enabledExtensionCount=static_cast<uint32_t>(vrsExts.size());ci.ppEnabledExtensionNames=vrsExts.data();
        if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}
        VK_CHECK(vkCreateDevice(physicalDevice_,&ci,nullptr,&device_));
        vkGetDeviceQueue(device_,queueIndices_.graphicsFamily.value(),0,&graphicsQueue_);
        vkGetDeviceQueue(device_,queueIndices_.presentFamily.value(),0,&presentQueue_);
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
    {
        static auto start = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now() - start).count();

        VkCommandBufferBeginInfo bi{};bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd,&bi));

        // ── 管线级别 VRS：设置当前绘制命令的着色率 ──────────────────────
        if (fpSetShadingRate_) {
            // 根据时间动态切换着色率（演示效果）
            float cycleTime = fmod(time, 6.0f);
            VkExtent2D shadingRate;
            if (cycleTime < 2.0f) {
                shadingRate = {1, 1};  // 全精度
                glfwSetWindowTitle(window_, "VRS 1x1（全精度）");
            } else if (cycleTime < 4.0f) {
                shadingRate = {2, 2};  // 4像素共享一次着色
                glfwSetWindowTitle(window_, "VRS 2x2（25%片段着色器调用）");
            } else {
                shadingRate = {4, 4};  // 16像素共享一次着色
                glfwSetWindowTitle(window_, "VRS 4x4（6.25%片段着色器调用！）");
            }

            // combinerOps：如何合并管线率/图元率/附件率
            VkFragmentShadingRateCombinerOpKHR combinerOps[2] = {
                VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,   // pipeline rate 优先
                VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
            };
            fpSetShadingRate_(cmd, &shadingRate, combinerOps);
        }

        VkClearValue clear{};clear.color.float32[0]=0.02f;clear.color.float32[2]=0.05f;clear.color.float32[3]=1.0f;
        VkRenderPassBeginInfo rp{};rp.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;rp.renderPass=renderPass_;rp.framebuffer=framebuffers_[imageIndex];rp.renderArea={{0,0},swapchainExtent_};rp.clearValueCount=1;rp.pClearValues=&clear;
        vkCmdBeginRenderPass(cmd,&rp,VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline_);
        VkViewport vp{0,0,(float)swapchainExtent_.width,(float)swapchainExtent_.height,0,1};vkCmdSetViewport(cmd,0,1,&vp);
        VkRect2D sc{{0,0},swapchainExtent_};vkCmdSetScissor(cmd,0,1,&sc);
        VRSPushConstants pc{};
        pc.resolution[0]=(float)swapchainExtent_.width;pc.resolution[1]=(float)swapchainExtent_.height;pc.time=time;
        vkCmdPushConstants(cmd,pipelineLayout_,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(pc),&pc);
        vkCmdDraw(cmd,3,1,0,0);  // 全屏三角形
        vkCmdEndRenderPass(cmd);
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
        std::cout<<"🎯 每2秒切换一次着色率：1x1→2x2→4x4（观察窗口标题 + 质量变化）\n";
        while(!glfwWindowShouldClose(window_)){glfwPollEvents();if(glfwGetKey(window_,GLFW_KEY_ESCAPE)==GLFW_PRESS)glfwSetWindowShouldClose(window_,GLFW_TRUE);drawFrame();}
        vkDeviceWaitIdle(device_);
    }

    void printVRSGuide()
    {
        std::cout<<"\n════════════════════════════════════════\n";
        std::cout<<" Variable Rate Shading (VRS) 概念速查\n";
        std::cout<<"════════════════════════════════════════\n\n";
        std::cout<<"1. 扩展：VK_KHR_fragment_shading_rate\n\n";
        std::cout<<"2. 三种级别（组合顺序）：\n";
        std::cout<<"   管线率  → 图元率 → 附件率\n";
        std::cout<<"   Combiner Op 决定如何合并\n\n";
        std::cout<<"3. 着色率选项：\n";
        std::cout<<"   1x1 (100%) → 2x1 (50%) → 1x2 (50%)\n";
        std::cout<<"   → 2x2 (25%) → 4x2 (12.5%) → 4x4 (6.25%)\n\n";
        std::cout<<"4. API：vkCmdSetFragmentShadingRateKHR()\n\n";
        std::cout<<"5. 最佳实践：\n";
        std::cout<<"   - VR: 眼球追踪 → 注视点 1x1，周边 4x4\n";
        std::cout<<"   - 运动模糊区域：2x2 或 4x4\n";
        std::cout<<"   - 后处理阶段：4x4（节省 94% 片段调用）\n";
    }

    void createGraphicsPipeline()
    {
        VkShaderModule vert=createShaderModuleFromFile(device_,"vrs_demo.vert.spv");
        VkShaderModule frag=createShaderModuleFromFile(device_,"vrs_demo.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vert,"main",nullptr},{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,frag,"main",nullptr}};
        VkPipelineVertexInputStateCreateInfo vi{};vi.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,nullptr,0,VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,nullptr,0,1,nullptr,1,nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};rs.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;rs.polygonMode=VK_POLYGON_MODE_FILL;rs.lineWidth=1.0f;rs.cullMode=VK_CULL_MODE_NONE;
        VkPipelineMultisampleStateCreateInfo ms{};ms.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};cba.colorWriteMask=0xf;
        VkPipelineColorBlendStateCreateInfo cb{};cb.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;cb.attachmentCount=1;cb.pAttachments=&cba;
        std::vector<VkDynamicState> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR,VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR};  // ← VRS 也是动态状态！
        VkPipelineDynamicStateCreateInfo dynS{};dynS.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;dynS.dynamicStateCount=static_cast<uint32_t>(dyn.size());dynS.pDynamicStates=dyn.data();
        VkPushConstantRange pcRange{};pcRange.stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT;pcRange.offset=0;pcRange.size=sizeof(VRSPushConstants);
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_,&pli,nullptr,&pipelineLayout_));
        VkGraphicsPipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;pi.stageCount=2;pi.pStages=stages;pi.pVertexInputState=&vi;pi.pInputAssemblyState=&ia;pi.pViewportState=&vs;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pColorBlendState=&cb;pi.pDynamicState=&dynS;pi.layout=pipelineLayout_;pi.renderPass=renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&pipeline_));
        vkDestroyShaderModule(device_,frag,nullptr);vkDestroyShaderModule(device_,vert,nullptr);
        std::cout<<"✅ VRS 管线创建（VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR）\n";
    }
    void createInstance(){VkApplicationInfo ai{};ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;ai.apiVersion=VK_API_VERSION_1_3;auto exts=getRequiredInstanceExtensions();VkInstanceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ci.pApplicationInfo=&ai;ci.enabledExtensionCount=static_cast<uint32_t>(exts.size());ci.ppEnabledExtensionNames=exts.data();ci.flags|=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}VK_CHECK(vkCreateInstance(&ci,nullptr,&instance_));}
    void createSurface(){VK_CHECK(glfwCreateWindowSurface(instance_,window_,nullptr,&surface_));}
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
        for(int i=0;i<MAX_FRAMES;++i){vkDestroySemaphore(device_,imageAvailableSems_[i],nullptr);vkDestroySemaphore(device_,renderFinishedSems_[i],nullptr);vkDestroyFence(device_,inFlightFences_[i],nullptr);}
        vkDestroyCommandPool(device_,commandPool_,nullptr);for(auto&fb:framebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);
        vkDestroyPipeline(device_,pipeline_,nullptr);vkDestroyPipelineLayout(device_,pipelineLayout_,nullptr);vkDestroyRenderPass(device_,renderPass_,nullptr);
        for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        vkDestroyDevice(device_,nullptr);vkDestroySurfaceKHR(instance_,surface_,nullptr);vkDestroyInstance(instance_,nullptr);glfwDestroyWindow(window_);glfwTerminate();
        std::cout<<"✅ 清理完成。\n";
    }
};

int main()
{
    std::cout<<"═══════════════════════════════════════════════════════════════════\n";
    std::cout<<" 第29章：Variable Rate Shading（VRS / 可变着色率）\n";
    std::cout<<"\n";
    std::cout<<" 扩展：VK_KHR_fragment_shading_rate\n";
    std::cout<<"\n";
    std::cout<<" 演示：每2秒自动切换着色率\n";
    std::cout<<"   1x1 → 100%  片段着色器调用（全精度）\n";
    std::cout<<"   2x2 → 25%   片段着色器调用（轻微模糊）\n";
    std::cout<<"   4x4 → 6.25% 片段着色器调用（明显模糊）\n";
    std::cout<<"\n";
    std::cout<<" 关键 API：\n";
    std::cout<<"   vkCmdSetFragmentShadingRateKHR(cmd, &rate, combinerOps)\n";
    std::cout<<"   VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR（管线动态状态）\n";
    std::cout<<"═══════════════════════════════════════════════════════════════════\n\n";
    Ch29App app;
    try{app.run();}catch(const std::exception&e){std::cerr<<"❌ "<<e.what()<<"\n";return 1;}
    return 0;
}
