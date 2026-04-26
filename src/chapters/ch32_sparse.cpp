/**
 * @file ch32_sparse.cpp
 * @brief 第32章：Sparse Resources（稀疏资源 / 虚拟纹理）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【什么是 Sparse Resources？】
 *
 *  传统纹理：创建时必须为完整纹理分配显存
 *    1GB 纹理 = 必须占用 1GB 显存（即使只用到其中 10%）
 *
 *  Sparse Resources（稀疏资源）：
 *    虚拟地址空间 >> 实际物理显存
 *    只需为实际访问到的"瓦片"（Tile）分配物理内存
 *    未使用的区域不占显存！
 *
 * 【核心概念】
 *
 *  ┌──────────────────────────────────────────────────────────────────┐
 *  │  虚拟纹理（16K×16K = 256MB）                                     │
 *  │                                                                  │
 *  │  [已分配] [已分配] [未分配] [未分配] [未分配]                   │
 *  │  [已分配] [已分配] [已分配] [未分配] [未分配]                   │
 *  │  [未分配] [未分配] [已分配] [未分配] [未分配]                   │
 *  │                   ↑                                             │
 *  │             只有这几块瓦片有实际物理内存（如 16MB）              │
 *  └──────────────────────────────────────────────────────────────────┘
 *
 * 【应用场景】
 *
 *  1. 地形地貌纹理（Virtual Texturing / Mega-Textures）
 *     - 整个游戏世界用一张超大纹理
 *     - 只加载玩家附近可见的瓦片
 *
 *  2. 大型 3D 模型纹理
 *     - 只加载相机朝向的表面纹理
 *
 *  3. GPU 内存超额使用
 *     - 总虚拟纹理 > GPU 显存容量
 *     - 流式加载/卸载瓦片
 *
 * 【Vulkan Sparse Binding API】
 *
 *  1. 创建 Sparse Image（VK_IMAGE_CREATE_SPARSE_BINDING_BIT）
 *     不立即分配内存
 *
 *  2. 查询瓦片信息
 *     vkGetImageSparseMemoryRequirements → 获取瓦片大小、布局
 *
 *  3. 按需分配/绑定物理内存
 *     vkQueueBindSparse → 将物理内存绑定到虚拟地址范围
 *     （可以随时绑定/解绑，实现流式加载）
 *
 *  4. 着色器透明访问
 *     着色器不需要知道稀疏性，正常采样
 *     未绑定的区域行为未定义（可设置"残差"颜色）
 *
 * 【瓦片（Tile）属性】
 *
 *  VkSparseImageMemoryRequirements：
 *    imageGranularity: 瓦片大小（如 128×128 像素）
 *    mipTailFirstLod:  小 mip 层合并为"mip tail"的起始层
 *    flags:            是否需要单独绑定深度和模板
 *
 * 【本章演示】
 *
 *  - 创建 4K×4K 稀疏纹理
 *  - 只为中央 4×4 瓦片区域分配物理内存
 *  - 每帧动态绑定/解绑不同瓦片（流式加载模拟）
 *  - 可视化哪些瓦片已分配（绿色）vs 未分配（黑色）
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES  = 2;

// 稀疏纹理尺寸（虚拟地址空间）
constexpr uint32_t SPARSE_TEX_SIZE = 4096;  // 4K × 4K

class Ch32App {
public:
    void run()
    {
        initWindow();
        if (!initVulkan()) { printSparseGuide(); glfwDestroyWindow(window_); glfwTerminate(); return; }
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
    VkQueue          sparseQueue_    = VK_NULL_HANDLE;   // 专用 sparse binding 队列
    VkSwapchainKHR   swapchain_      = VK_NULL_HANDLE;
    VkRenderPass     renderPass_     = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline       pipeline_       = VK_NULL_HANDLE;
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;
    VkDescriptorPool descPool_       = VK_NULL_HANDLE;
    VkDescriptorSet  descSet_        = VK_NULL_HANDLE;
    VkSampler        sampler_        = VK_NULL_HANDLE;

    // ─── 稀疏纹理相关 ────────────────────────────────────────────────────
    VkImage        sparseImage_      = VK_NULL_HANDLE;
    VkImageView    sparseView_       = VK_NULL_HANDLE;
    VkExtent3D     tileSize_{};      ///< 单个瓦片的像素尺寸
    uint32_t       numTilesX_  = 0;  ///< 水平瓦片数
    uint32_t       numTilesY_  = 0;  ///< 垂直瓦片数

    // 已分配的物理内存（每个活跃瓦片一块）
    struct TileMemory {
        VkDeviceMemory memory;
        uint32_t       tileX, tileY;
    };
    std::vector<TileMemory> allocatedTiles_;
    uint32_t sparseMemTypeIndex_ = 0;

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
    uint32_t currentFrame_  = 0;
    uint64_t frameCount_    = 0;
    bool     resized_       = false;

    void initWindow()
    {
        glfwInit(); glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT,
            "Ch32 - Sparse Resources（虚拟纹理/稀疏绑定）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_,
            [](GLFWwindow*w,int,int){
                reinterpret_cast<Ch32App*>(glfwGetWindowUserPointer(w))->resized_=true;
            });
    }

    bool initVulkan()
    {
        try {
            createInstance(); createSurface();
            if (!pickPhysicalDeviceSparse()) return false;
            createLogicalDeviceSparse();
            createSwapchain(); createImageViews(); createRenderPass();
            createDescriptorSetLayout();
            createGraphicsPipeline();
            createFramebuffers(); createCommandPool();
            createSampler();
            createSparseTexture();          // ← 核心：创建稀疏图像
            initialTileBinding();           // ← 绑定初始瓦片
            createDescriptorSet();
            createCommandBuffers(); createSyncObjects();
            std::cout << "\n✅ Sparse Resources 初始化完成！\n";
            std::cout << "📐 稀疏纹理：" << SPARSE_TEX_SIZE << "×" << SPARSE_TEX_SIZE << "\n";
            std::cout << "🧩 瓦片尺寸：" << tileSize_.width << "×" << tileSize_.height << "\n";
            std::cout << "📊 虚拟瓦片总数：" << numTilesX_ << "×" << numTilesY_ << "="
                      << numTilesX_*numTilesY_ << "\n";
        } catch (const std::exception& e) {
            std::cerr << "⚠️  " << e.what() << "\n"; return false;
        }
        return true;
    }

    bool pickPhysicalDeviceSparse()
    {
        uint32_t c=0; vkEnumeratePhysicalDevices(instance_,&c,nullptr);
        std::vector<VkPhysicalDevice> devs(c);
        vkEnumeratePhysicalDevices(instance_,&c,devs.data());
        for (auto& d : devs) {
            if (!findQueueFamilies(d,surface_).isComplete()) continue;
            VkPhysicalDeviceFeatures feat;
            vkGetPhysicalDeviceFeatures(d, &feat);
            if (!feat.sparseBinding || !feat.sparseResidencyImage2D) {
                std::cout<<"⚠️  GPU 不支持 sparseBinding/sparseResidencyImage2D\n"; continue;
            }
            physicalDevice_=d;
            VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d,&p);
            std::cout<<"✅ GPU: "<<p.deviceName<<" (Sparse Binding 支持)\n";

            // 查询稀疏绑定队列
            uint32_t qfCount=0; vkGetPhysicalDeviceQueueFamilyProperties(d,&qfCount,nullptr);
            std::vector<VkQueueFamilyProperties> qfs(qfCount);
            vkGetPhysicalDeviceQueueFamilyProperties(d,&qfCount,qfs.data());
            for(uint32_t i=0;i<qfCount;++i)
                if(qfs[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)
                    { std::cout<<"  Sparse Queue 族：" << i << "\n"; break; }
            return true;
        }
        return false;
    }

    void createLogicalDeviceSparse()
    {
        queueIndices_=findQueueFamilies(physicalDevice_,surface_);
        std::set<uint32_t> fams={queueIndices_.graphicsFamily.value(),queueIndices_.presentFamily.value()};
        const float pri=1.0f; std::vector<VkDeviceQueueCreateInfo> qcis;
        for(uint32_t f:fams){VkDeviceQueueCreateInfo q{};q.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;q.queueFamilyIndex=f;q.queueCount=1;q.pQueuePriorities=&pri;qcis.push_back(q);}

        VkPhysicalDeviceFeatures feat{};
        feat.samplerAnisotropy       = VK_TRUE;
        feat.sparseBinding           = VK_TRUE;   // ← 必须启用！
        feat.sparseResidencyImage2D  = VK_TRUE;   // ← 稀疏 2D 图像

        VkDeviceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount=static_cast<uint32_t>(qcis.size());ci.pQueueCreateInfos=qcis.data();
        ci.pEnabledFeatures=&feat;
        ci.enabledExtensionCount=static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        ci.ppEnabledExtensionNames=DEVICE_EXTENSIONS.data();
        if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}
        VK_CHECK(vkCreateDevice(physicalDevice_,&ci,nullptr,&device_));
        vkGetDeviceQueue(device_,queueIndices_.graphicsFamily.value(),0,&graphicsQueue_);
        vkGetDeviceQueue(device_,queueIndices_.presentFamily.value(),0,&presentQueue_);
        sparseQueue_ = graphicsQueue_;   // 简化：复用图形队列
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 创建稀疏图像
    // ═══════════════════════════════════════════════════════════════════════

    void createSparseTexture()
    {
        // ① 创建稀疏图像（不分配内存！）
        VkImageCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType     = VK_IMAGE_TYPE_2D;
        ci.extent        = {SPARSE_TEX_SIZE, SPARSE_TEX_SIZE, 1};
        ci.mipLevels     = 1; ci.arrayLayers = 1;
        ci.format        = VK_FORMAT_R8G8B8A8_SRGB;
        ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage         = VK_IMAGE_USAGE_SAMPLED_BIT;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        // 关键标志：稀疏绑定（虚拟地址空间，无需立即分配内存）
        ci.flags         = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                           VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;  // 允许部分绑定
        VK_CHECK(vkCreateImage(device_, &ci, nullptr, &sparseImage_));

        // ② 查询稀疏内存需求（瓦片大小等）
        uint32_t sparseReqCount = 0;
        vkGetImageSparseMemoryRequirements(device_, sparseImage_, &sparseReqCount, nullptr);
        std::vector<VkSparseImageMemoryRequirements> sparseReqs(sparseReqCount);
        vkGetImageSparseMemoryRequirements(device_, sparseImage_, &sparseReqCount, sparseReqs.data());

        if (sparseReqs.empty()) throw std::runtime_error("无稀疏内存需求信息");

        // 找到 mip level 0 的稀疏需求
        VkSparseImageMemoryRequirements& sparseReq = sparseReqs[0];
        tileSize_ = sparseReq.formatProperties.imageGranularity;  // 瓦片尺寸（如 128×128）
        numTilesX_ = (SPARSE_TEX_SIZE + tileSize_.width  - 1) / tileSize_.width;
        numTilesY_ = (SPARSE_TEX_SIZE + tileSize_.height - 1) / tileSize_.height;

        std::cout << "🧩 瓦片信息：\n";
        std::cout << "   虚拟瓦片：" << numTilesX_ << "×" << numTilesY_ << "\n";
        std::cout << "   瓦片内存：" << tileSize_.width * tileSize_.height * 4 / 1024 << " KB/瓦片\n";

        // ③ 查询稀疏图像对应的内存类型
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device_, sparseImage_, &memReqs);
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReqs.memoryTypeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                sparseMemTypeIndex_ = i; break;
            }
        }

        // ④ 创建 ImageView
        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = sparseImage_; vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = VK_FORMAT_R8G8B8A8_SRGB;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &sparseView_));

        std::cout << "✅ 稀疏图像创建完成（虚拟内存 = "
                  << SPARSE_TEX_SIZE*SPARSE_TEX_SIZE*4/1024/1024 << " MB"
                  << "，实际物理内存 = 0 MB）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 绑定瓦片（核心：vkQueueBindSparse）
    // ═══════════════════════════════════════════════════════════════════════

    void bindTile(uint32_t tileX, uint32_t tileY, VkDeviceMemory memory)
    {
        // 指定要绑定的内存区域
        VkSparseImageMemoryBind bind{};
        bind.subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bind.subresource.mipLevel   = 0;
        bind.subresource.arrayLayer = 0;
        bind.offset   = {static_cast<int32_t>(tileX * tileSize_.width),
                          static_cast<int32_t>(tileY * tileSize_.height), 0};
        bind.extent   = {std::min(tileSize_.width,  SPARSE_TEX_SIZE - bind.offset.x),
                          std::min(tileSize_.height, SPARSE_TEX_SIZE - bind.offset.y), 1};
        bind.memory   = memory;  // 绑定实际物理内存
        bind.memoryOffset = 0;
        bind.flags    = 0;

        VkSparseImageMemoryBindInfo bindInfo{};
        bindInfo.image     = sparseImage_;
        bindInfo.bindCount = 1;
        bindInfo.pBinds    = &bind;

        VkBindSparseInfo sparseBindInfo{};
        sparseBindInfo.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
        sparseBindInfo.imageBindCount = 1;
        sparseBindInfo.pImageBinds    = &bindInfo;

        // vkQueueBindSparse：异步绑定（不需要停止渲染！）
        VK_CHECK(vkQueueBindSparse(sparseQueue_, 1, &sparseBindInfo, VK_NULL_HANDLE));
        vkQueueWaitIdle(sparseQueue_);
    }

    void unbindTile(uint32_t tileX, uint32_t tileY)
    {
        VkSparseImageMemoryBind bind{};
        bind.subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bind.offset  = {static_cast<int32_t>(tileX * tileSize_.width),
                         static_cast<int32_t>(tileY * tileSize_.height), 0};
        bind.extent  = {tileSize_.width, tileSize_.height, 1};
        bind.memory  = VK_NULL_HANDLE;   // 解绑：memory = NULL
        bind.flags   = 0;

        VkSparseImageMemoryBindInfo bindInfo{sparseImage_, 1, &bind};
        VkBindSparseInfo sparseBindInfo{};
        sparseBindInfo.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
        sparseBindInfo.imageBindCount = 1;
        sparseBindInfo.pImageBinds = &bindInfo;
        VK_CHECK(vkQueueBindSparse(sparseQueue_, 1, &sparseBindInfo, VK_NULL_HANDLE));
        vkQueueWaitIdle(sparseQueue_);
    }

    VkDeviceMemory allocateTileMemory()
    {
        VkDeviceSize tileMemSize = tileSize_.width * tileSize_.height * 4;   // RGBA8
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = tileMemSize;
        ai.memoryTypeIndex = sparseMemTypeIndex_;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &mem));
        return mem;
    }

    void initialTileBinding()
    {
        // 绑定中央 4×4 区域的瓦片（模拟相机附近已加载的区域）
        uint32_t cx = numTilesX_ / 2 - 2;
        uint32_t cy = numTilesY_ / 2 - 2;
        for (uint32_t ty = cy; ty < cy+4 && ty < numTilesY_; ++ty) {
            for (uint32_t tx = cx; tx < cx+4 && tx < numTilesX_; ++tx) {
                VkDeviceMemory mem = allocateTileMemory();
                bindTile(tx, ty, mem);
                allocatedTiles_.push_back({mem, tx, ty});
                std::cout << "  绑定瓦片 (" << tx << "," << ty << ")\n";
            }
        }
        std::cout << "✅ 已绑定 " << allocatedTiles_.size() << " 个瓦片 ("
                  << allocatedTiles_.size() * tileSize_.width * tileSize_.height * 4 / 1024
                  << " KB 物理内存)\n";
    }

    void dynamicTileStreaming()
    {
        // 每 120 帧演示一次流式加载/卸载
        if (frameCount_ % 120 != 0) return;

        // 解绑所有现有瓦片
        for (auto& tile : allocatedTiles_) {
            unbindTile(tile.tileX, tile.tileY);
            vkFreeMemory(device_, tile.memory, nullptr);
        }
        allocatedTiles_.clear();

        // 随机绑定新的 4×4 区域（模拟玩家移动）
        static uint32_t cx = 0, cy = 0;
        cx = (cx + 5) % (numTilesX_ - 4);
        cy = (cy + 3) % (numTilesY_ - 4);

        for (uint32_t ty = cy; ty < cy+4 && ty < numTilesY_; ++ty) {
            for (uint32_t tx = cx; tx < cx+4 && tx < numTilesX_; ++tx) {
                VkDeviceMemory mem = allocateTileMemory();
                bindTile(tx, ty, mem);
                allocatedTiles_.push_back({mem, tx, ty});
            }
        }
        std::cout << "🔄 流式加载：瓦片区域移至 (" << cx << "," << cy << ")\n";
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo bi{};bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd,&bi));
        VkClearValue clear{};clear.color.float32[0]=0.02f;clear.color.float32[2]=0.05f;clear.color.float32[3]=1.0f;
        VkRenderPassBeginInfo rp{};rp.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;rp.renderPass=renderPass_;rp.framebuffer=framebuffers_[imageIndex];rp.renderArea={{0,0},swapchainExtent_};rp.clearValueCount=1;rp.pClearValues=&clear;
        vkCmdBeginRenderPass(cmd,&rp,VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline_);
        VkViewport vp{0,0,(float)swapchainExtent_.width,(float)swapchainExtent_.height,0,1};vkCmdSetViewport(cmd,0,1,&vp);
        VkRect2D sc{{0,0},swapchainExtent_};vkCmdSetScissor(cmd,0,1,&sc);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipelineLayout_,0,1,&descSet_,0,nullptr);
        vkCmdDraw(cmd,3,1,0,0);   // 全屏三角形显示稀疏纹理
        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame()
    {
        vkWaitForFences(device_,1,&inFlightFences_[currentFrame_],VK_TRUE,UINT64_MAX);
        uint32_t imgIdx=0;VkResult r=vkAcquireNextImageKHR(device_,swapchain_,UINT64_MAX,imageAvailableSems_[currentFrame_],VK_NULL_HANDLE,&imgIdx);
        if(r==VK_ERROR_OUT_OF_DATE_KHR){recreateSwapchain();return;}
        dynamicTileStreaming();  // 模拟流式加载
        vkResetFences(device_,1,&inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_],0);
        recordCommandBuffer(commandBuffers_[currentFrame_],imgIdx);
        VkSemaphore ws[]={imageAvailableSems_[currentFrame_]};VkPipelineStageFlags wst[]={VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};VkSemaphore ss[]={renderFinishedSems_[currentFrame_]};
        VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;si.waitSemaphoreCount=1;si.pWaitSemaphores=ws;si.pWaitDstStageMask=wst;si.commandBufferCount=1;si.pCommandBuffers=&commandBuffers_[currentFrame_];si.signalSemaphoreCount=1;si.pSignalSemaphores=ss;
        VK_CHECK(vkQueueSubmit(graphicsQueue_,1,&si,inFlightFences_[currentFrame_]));
        VkSwapchainKHR scs[]={swapchain_};VkPresentInfoKHR pi{};pi.sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;pi.waitSemaphoreCount=1;pi.pWaitSemaphores=ss;pi.swapchainCount=1;pi.pSwapchains=scs;pi.pImageIndices=&imgIdx;
        r=vkQueuePresentKHR(presentQueue_,&pi);
        if(r==VK_ERROR_OUT_OF_DATE_KHR||r==VK_SUBOPTIMAL_KHR||resized_){resized_=false;recreateSwapchain();}
        currentFrame_=(currentFrame_+1)%MAX_FRAMES;++frameCount_;
    }

    void mainLoop()
    {
        std::cout<<"📜 稀疏纹理演示（每120帧流式加载新瓦片，ESC退出）...\n";
        while(!glfwWindowShouldClose(window_)){glfwPollEvents();if(glfwGetKey(window_,GLFW_KEY_ESCAPE)==GLFW_PRESS)glfwSetWindowShouldClose(window_,GLFW_TRUE);drawFrame();}
        vkDeviceWaitIdle(device_);
    }

    void printSparseGuide()
    {
        std::cout<<"\n Sparse Resources（稀疏资源）概念速查\n\n";
        std::cout<<"创建：\n";
        std::cout<<"  VkImageCreateFlags: SPARSE_BINDING_BIT | SPARSE_RESIDENCY_BIT\n";
        std::cout<<"  不传递 VkMemoryRequirements → 不立即分配内存！\n\n";
        std::cout<<"查询瓦片信息：\n";
        std::cout<<"  vkGetImageSparseMemoryRequirements → 获取瓦片大小/布局\n\n";
        std::cout<<"绑定/解绑：\n";
        std::cout<<"  VkSparseImageMemoryBind: image区域 ↔ 物理内存\n";
        std::cout<<"  vkQueueBindSparse → 提交绑定操作（异步！）\n\n";
        std::cout<<"应用：\n";
        std::cout<<"  Virtual Texturing（虚拟纹理）\n";
        std::cout<<"  Mega-Textures（地形超大纹理）\n";
        std::cout<<"  GPU 内存超额使用（Texture Streaming）\n";
    }

    // ─── 辅助代码 ─────────────────────────────────────────────────────────────

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding b{1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;ci.bindingCount=1;ci.pBindings=&b;
        VK_CHECK(vkCreateDescriptorSetLayout(device_,&ci,nullptr,&setLayout_));
    }
    void createGraphicsPipeline()
    {
        VkShaderModule vert=createShaderModuleFromFile(device_,"tonemap.vert.spv");
        VkShaderModule frag=createShaderModuleFromFile(device_,"vrs_demo.frag.spv");  // 可视化纹理
        VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vert,"main",nullptr},{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,frag,"main",nullptr}};
        VkPipelineVertexInputStateCreateInfo vi{};vi.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,nullptr,0,VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,nullptr,0,1,nullptr,1,nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};rs.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;rs.polygonMode=VK_POLYGON_MODE_FILL;rs.lineWidth=1.0f;rs.cullMode=VK_CULL_MODE_NONE;
        VkPipelineMultisampleStateCreateInfo ms{};ms.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};cba.colorWriteMask=0xf;
        VkPipelineColorBlendStateCreateInfo cb{};cb.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;cb.attachmentCount=1;cb.pAttachments=&cba;
        std::vector<VkDynamicState> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};dynS.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;dynS.dynamicStateCount=static_cast<uint32_t>(dyn.size());dynS.pDynamicStates=dyn.data();
        // Push constants for VRS demo shader
        VkPushConstantRange pcRange{};pcRange.stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT;pcRange.size=12;
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.setLayoutCount=1;pli.pSetLayouts=&setLayout_;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_,&pli,nullptr,&pipelineLayout_));
        VkGraphicsPipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;pi.stageCount=2;pi.pStages=stages;pi.pVertexInputState=&vi;pi.pInputAssemblyState=&ia;pi.pViewportState=&vs;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pColorBlendState=&cb;pi.pDynamicState=&dynS;pi.layout=pipelineLayout_;pi.renderPass=renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&pipeline_));
        vkDestroyShaderModule(device_,frag,nullptr);vkDestroyShaderModule(device_,vert,nullptr);
    }
    void createSampler(){VkSamplerCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;ci.magFilter=VK_FILTER_LINEAR;ci.minFilter=VK_FILTER_LINEAR;ci.addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;ci.addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;VK_CHECK(vkCreateSampler(device_,&ci,nullptr,&sampler_));}
    void createDescriptorSet()
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1};
        VkDescriptorPoolCreateInfo poolCI{};poolCI.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;poolCI.poolSizeCount=1;poolCI.pPoolSizes=&ps;poolCI.maxSets=1;
        VK_CHECK(vkCreateDescriptorPool(device_,&poolCI,nullptr,&descPool_));
        VkDescriptorSetAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;ai.descriptorPool=descPool_;ai.descriptorSetCount=1;ai.pSetLayouts=&setLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_,&ai,&descSet_));
        VkDescriptorImageInfo ii{};ii.sampler=sampler_;ii.imageView=sparseView_;ii.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w.dstSet=descSet_;w.dstBinding=1;w.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;w.descriptorCount=1;w.pImageInfo=&ii;
        vkUpdateDescriptorSets(device_,1,&w,0,nullptr);
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
        for(auto&t:allocatedTiles_){unbindTile(t.tileX,t.tileY);vkFreeMemory(device_,t.memory,nullptr);}
        vkDestroySampler(device_,sampler_,nullptr);vkDestroyImageView(device_,sparseView_,nullptr);vkDestroyImage(device_,sparseImage_,nullptr);
        vkDestroyDescriptorPool(device_,descPool_,nullptr);vkDestroyDescriptorSetLayout(device_,setLayout_,nullptr);
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
    std::cout<<" 第32章：Sparse Resources（稀疏资源 / 虚拟纹理）\n";
    std::cout<<"\n";
    std::cout<<" 核心特性：VK_IMAGE_CREATE_SPARSE_BINDING_BIT\n";
    std::cout<<"\n";
    std::cout<<" 关键 API：\n";
    std::cout<<"   vkGetImageSparseMemoryRequirements → 获取瓦片信息\n";
    std::cout<<"   vkQueueBindSparse → 动态绑定/解绑物理内存（异步）\n";
    std::cout<<"   VkSparseImageMemoryBind.memory = VK_NULL_HANDLE → 解绑\n";
    std::cout<<"\n";
    std::cout<<" 演示：4K×4K 稀疏纹理，每120帧流式加载新瓦片区域\n";
    std::cout<<"═══════════════════════════════════════════════════════════════════\n\n";
    Ch32App app;
    try{app.run();}catch(const std::exception&e){std::cerr<<"❌ "<<e.what()<<"\n";return 1;}
    return 0;
}
