/**
 * @file ch30_vma.cpp
 * @brief 第30章：Vulkan Memory Allocator (VMA) + 高级内存管理
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【为什么需要 VMA？】
 *
 *  原生 Vulkan 内存管理的痛点：
 *
 *  1. vkAllocateMemory 调用次数有上限！
 *     maxMemoryAllocationCount（通常 4096 次）
 *     → 每个缓冲/图像单独分配内存很快耗尽
 *
 *  2. 内存对齐复杂：
 *     不同资源的 alignment 要求各不相同（16/256/4096 字节等）
 *
 *  3. 内存碎片：
 *     频繁分配/释放导致大量小碎片，无法分配大块内存
 *
 *  VMA 解决方案：
 *  - 内存池（Memory Pool）：预分配大块内存，内部切割
 *  - 自动处理对齐：计算每个资源的实际偏移
 *  - 碎片整理（Defragmentation）：移动资源，合并碎片
 *  - 统计分析：报告内存使用情况
 *
 * 【VMA 核心 API】
 *
 *  // 初始化
 *  VmaAllocatorCreateInfo createInfo{};
 *  vmaCreateAllocator(&createInfo, &allocator);
 *
 *  // 创建缓冲（自动分配内存！）
 *  VmaAllocationCreateInfo allocCI{};
 *  allocCI.usage = VMA_MEMORY_USAGE_GPU_ONLY;
 *  vmaCreateBuffer(allocator, &bufferCI, &allocCI, &buffer, &allocation, nullptr);
 *
 *  // 映射内存（自动处理持久化/临时映射）
 *  void* mapped;
 *  vmaMapMemory(allocator, allocation, &mapped);
 *  vmaUnmapMemory(allocator, allocation);
 *
 *  // 销毁
 *  vmaDestroyBuffer(allocator, buffer, allocation);
 *  vmaDestroyAllocator(allocator);
 *
 * 【VMA 内存使用类型】
 *
 *  VMA_MEMORY_USAGE_GPU_ONLY    → Device Local（最快，CPU不能访问）
 *  VMA_MEMORY_USAGE_CPU_ONLY    → Host Visible + Coherent（CPU读写）
 *  VMA_MEMORY_USAGE_CPU_TO_GPU  → Upload Staging Buffer
 *  VMA_MEMORY_USAGE_GPU_TO_CPU  → Readback Buffer
 *  VMA_MEMORY_USAGE_CPU_COPY    → 传输用途
 *  VMA_MEMORY_USAGE_AUTO        → 自动选择最佳类型（推荐）
 *
 * 【内存池（Custom Pools）】
 *
 *  对于频繁分配/释放的资源（如粒子、UI 元素），
 *  创建专用内存池避免碎片化：
 *
 *  VmaPoolCreateInfo poolCI{};
 *  poolCI.memoryTypeIndex = ...;
 *  poolCI.blockSize       = 64 * 1024 * 1024;  // 64MB 块
 *  vmaCreatePool(allocator, &poolCI, &pool);
 *
 * 【碎片整理】
 *
 *  vmaBeginDefragmentation / vmaEndDefragmentation
 *  在运行时移动资源，合并碎片空间
 *
 * 【本章示例（不依赖 VMA 库，教学版）】
 *
 *  实现一个简化版的 VMA：
 *  - VulkanMemoryPool 类：管理 VkDeviceMemory 内存块
 *  - 支持次分配（Sub-allocation）
 *  - 追踪碎片率和使用情况
 *  - 演示实际项目中的内存管理策略
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

// ─── 简化版 VMA：内存块管理器 ─────────────────────────────────────────────────

/**
 * @brief 内存分配记录
 */
struct Allocation {
    VkDeviceSize offset; ///< 在内存块中的偏移
    VkDeviceSize size;   ///< 分配大小
    bool isFree;         ///< 是否空闲
    std::string debugName;
};

/**
 * @brief 内存块（VkDeviceMemory 的包装）
 *
 * 每个 MemoryBlock 管理一个大的 VkDeviceMemory，
 * 通过自由列表（Free List）进行次分配。
 */
class MemoryBlock {
  public:
    static constexpr VkDeviceSize BLOCK_SIZE = 64 * 1024 * 1024; // 64 MB

    VkDevice device;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t memTypeIndex = 0;
    void* mappedPtr = nullptr;    // 持久化映射指针（如适用）
    std::list<Allocation> allocs; // 分配列表

    void init(VkDevice dev, uint32_t typeIdx, bool hostVisible) {
        device = dev;
        memTypeIndex = typeIdx;

        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = BLOCK_SIZE;
        ai.memoryTypeIndex = typeIdx;
        if (vkAllocateMemory(device, &ai, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("内存块分配失败");

        if (hostVisible)
            vkMapMemory(device, memory, 0, BLOCK_SIZE, 0, &mappedPtr);

        // 初始状态：整个块是一个大的空闲区域
        allocs.push_back({0, BLOCK_SIZE, true, "free"});
        std::cout << "  📦 新内存块：" << BLOCK_SIZE / 1024 / 1024 << " MB "
                  << "(类型=" << typeIdx << ")\n";
    }

    /**
     * @brief 在此块内分配指定大小的内存（首次适配）
     * @return 分配的偏移，失败返回 VK_WHOLE_SIZE
     */
    VkDeviceSize allocate(VkDeviceSize size, VkDeviceSize alignment, const std::string& name) {
        for (auto it = allocs.begin(); it != allocs.end(); ++it) {
            if (!it->isFree)
                continue;

            // 计算对齐后的偏移
            VkDeviceSize alignedOffset = (it->offset + alignment - 1) & ~(alignment - 1);
            VkDeviceSize padding = alignedOffset - it->offset;
            VkDeviceSize neededSize = padding + size;

            if (it->size < neededSize)
                continue;

            // 找到合适的空闲区域，切割它
            VkDeviceSize remainingSize = it->size - neededSize;

            it->isFree = false;
            it->offset = alignedOffset;
            it->size = size;
            it->debugName = name;

            if (padding > 0) {
                // 插入对齐填充块
                allocs.insert(it, {it->offset - padding, padding, true, "padding"});
            }
            if (remainingSize > 64) {
                // 剩余空间作为新的空闲块
                auto next = it;
                ++next;
                allocs.insert(next, {alignedOffset + size, remainingSize, true, "free"});
            }
            return alignedOffset;
        }
        return VK_WHOLE_SIZE; // 分配失败
    }

    void free(VkDeviceSize offset) {
        for (auto it = allocs.begin(); it != allocs.end(); ++it) {
            if (it->offset == offset && !it->isFree) {
                it->isFree = true;
                it->debugName = "free";
                mergeFreeBlocks();
                return;
            }
        }
    }

    // 合并相邻的空闲块（减少碎片）
    void mergeFreeBlocks() {
        for (auto it = allocs.begin(); it != allocs.end();) {
            auto next = it;
            ++next;
            if (next != allocs.end() && it->isFree && next->isFree) {
                it->size += next->size;
                allocs.erase(next);
            } else {
                ++it;
            }
        }
    }

    float getFragmentationRatio() const {
        VkDeviceSize freeTotal = 0, usedTotal = 0;
        uint32_t freeCount = 0;
        for (auto& a : allocs) {
            if (a.isFree) {
                freeTotal += a.size;
                ++freeCount;
            } else
                usedTotal += a.size;
        }
        if (freeTotal == 0)
            return 0.0f;
        // 碎片率：如果所有空闲空间都是碎片，返回1.0
        VkDeviceSize maxFreeBlock = 0;
        for (auto& a : allocs)
            if (a.isFree)
                maxFreeBlock = std::max(maxFreeBlock, a.size);
        return 1.0f - (float)maxFreeBlock / freeTotal;
    }

    void printStats() const {
        VkDeviceSize used = 0, free_ = 0;
        for (auto& a : allocs) {
            if (a.isFree)
                free_ += a.size;
            else
                used += a.size;
        }
        std::cout << "  内存块：已用=" << used / 1024 << "KB  空闲=" << free_ / 1024 << "KB  碎片率=" << std::fixed
                  << std::setprecision(1) << getFragmentationRatio() * 100 << "%\n";
        std::cout << "  分配列表（前10项）：\n";
        int count = 0;
        for (auto& a : allocs) {
            if (++count > 10) {
                std::cout << "    ...\n";
                break;
            }
            std::cout << "    [" << (a.isFree ? "FREE" : "USED") << "] " << a.offset / 1024 << "KB+" << a.size / 1024
                      << "KB"
                      << " (" << a.debugName << ")\n";
        }
    }

    void cleanup() {
        if (mappedPtr)
            vkUnmapMemory(device, memory);
        if (memory != VK_NULL_HANDLE)
            vkFreeMemory(device, memory, nullptr);
    }
};

/**
 * @brief 简化版内存分配器（演示 VMA 原理）
 */
class SimpleAllocator {
  public:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice_ = VK_NULL_HANDLE;

    struct SubAlloc {
        MemoryBlock* block;
        VkDeviceSize offset;
        VkDeviceMemory memory; // 方便绑定
    };

    std::vector<MemoryBlock*> blocks_;

    void init(VkDevice dev, VkPhysicalDevice phys) {
        device_ = dev;
        physDevice_ = phys;
        std::cout << "✅ SimpleAllocator 已初始化（演示 VMA 原理）\n";
    }

    uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(physDevice_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        throw std::runtime_error("找不到内存类型");
    }

    SubAlloc allocate(VkDeviceSize size,
                      uint32_t memTypeFilter,
                      VkMemoryPropertyFlags props,
                      VkDeviceSize alignment,
                      const std::string& name) {
        uint32_t typeIdx = findMemoryType(memTypeFilter, props);
        bool hostVisible = (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

        // 尝试在现有块中分配
        for (auto* block : blocks_) {
            if (block->memTypeIndex != typeIdx)
                continue;
            VkDeviceSize offset = block->allocate(size, alignment, name);
            if (offset != VK_WHOLE_SIZE) {
                std::cout << "  ✓ 在现有块中分配: " << name << " +" << size / 1024 << "KB\n";
                return {block, offset, block->memory};
            }
        }

        // 创建新块
        std::cout << "  + 创建新内存块（现有块不足）\n";
        auto* newBlock = new MemoryBlock();
        newBlock->init(device_, typeIdx, hostVisible);
        blocks_.push_back(newBlock);
        VkDeviceSize offset = newBlock->allocate(size, alignment, name);
        return {newBlock, offset, newBlock->memory};
    }

    void free(SubAlloc& alloc) {
        alloc.block->free(alloc.offset);
    }

    void printStats() {
        std::cout << "\n📊 内存分配器统计（共 " << blocks_.size() << " 个内存块）：\n";
        for (auto* b : blocks_)
            b->printStats();
    }

    void cleanup() {
        for (auto* b : blocks_) {
            b->cleanup();
            delete b;
        }
        blocks_.clear();
    }
};

// ─── 应用程序 ─────────────────────────────────────────────────────────────────

class Ch30App {
  public:
    void run() {
        initWindow();
        initVulkan();
        demonstrateMemoryManagement();
        mainLoop();
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
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;

    SimpleAllocator allocator_;

    // 使用 SimpleAllocator 分配的资源
    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    SimpleAllocator::SubAlloc vertexAlloc_{};
    VkBuffer uniformBuffer_ = VK_NULL_HANDLE;
    SimpleAllocator::SubAlloc uniformAlloc_{};

    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    QueueFamilyIndices queueIndices_;

    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(800, 600, "Ch30 - VMA + 内存管理（查看控制台输出）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch30App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        allocator_.init(device_, physicalDevice_); // ← 初始化自定义分配器
        createSwapchain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createResourcesWithAllocator(); // ← 使用自定义分配器
        createCommandBuffers();
        createSyncObjects();
        std::cout << "\n✅ 资源创建完成\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 演示内存分配/释放/碎片整理
    // ═══════════════════════════════════════════════════════════════════════

    void demonstrateMemoryManagement() {
        std::cout << "\n╔═════════════════════════════════════════════════════════╗\n";
        std::cout << "║  VMA 内存管理演示                                        ║\n";
        std::cout << "╠═════════════════════════════════════════════════════════╣\n";

        // ── 模拟分配多种资源 ──────────────────────────────────────────────
        std::cout << "\n【场景1：初始分配】\n";
        std::vector<std::pair<VkBuffer, SimpleAllocator::SubAlloc>> tempBuffers;

        const char* names[] = {"顶点缓冲A", "顶点缓冲B", "索引缓冲", "UBO", "SSBO", "纹理暂存"};
        uint32_t sizes_kb[] = {256, 512, 128, 4, 2048, 8192};
        for (int i = 0; i < 6; ++i) {
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = sizes_kb[i] * 1024;
            bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkBuffer buf;
            vkCreateBuffer(device_, &bci, nullptr, &buf);
            VkMemoryRequirements mr;
            vkGetBufferMemoryRequirements(device_, buf, &mr);
            auto alloc = allocator_.allocate(mr.size,
                                             mr.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                             mr.alignment,
                                             names[i]);
            vkBindBufferMemory(device_, buf, alloc.memory, alloc.offset);
            tempBuffers.push_back({buf, alloc});
        }
        allocator_.printStats();

        // ── 释放部分资源（制造碎片） ──────────────────────────────────────
        std::cout << "\n【场景2：释放部分资源（制造碎片）】\n";
        for (int i : {1, 3, 5}) { // 释放 B, UBO, 纹理暂存
            vkDestroyBuffer(device_, tempBuffers[i].first, nullptr);
            allocator_.free(tempBuffers[i].second);
            std::cout << "  释放: " << names[i] << "\n";
        }
        allocator_.printStats();

        // ── 清理临时缓冲 ──────────────────────────────────────────────────
        for (int i : {0, 2, 4}) {
            vkDestroyBuffer(device_, tempBuffers[i].first, nullptr);
            allocator_.free(tempBuffers[i].second);
        }

        std::cout << "\n【最佳实践总结】\n";
        std::cout << "  ✅ 使用内存池避免频繁 vkAllocateMemory\n";
        std::cout << "  ✅ 将相同类型的资源放入同一内存块\n";
        std::cout << "  ✅ 使用 VMA_MEMORY_USAGE_AUTO 让 VMA 选择最佳类型\n";
        std::cout << "  ✅ 定期调用 vmaDefragment 减少碎片\n";
        std::cout << "  ✅ 对动态资源使用 Persistent Mapping（永久映射）\n";

        std::cout << "\n【VMA 与手写内存管理对比】\n";
        std::cout << "  手写：需要自己实现 free list、对齐计算、碎片整理...\n";
        std::cout << "  VMA：一行代码：vmaCreateBuffer(allocator, &bufCI, &allocCI, ...)\n";
    }

    void createResourcesWithAllocator() {
        // 顶点缓冲（使用自定义分配器）
        struct SimpleVertex {
            float x, y, z, r, g, b;
        };
        SimpleVertex verts[] = {
            {0.0f, -0.5f, 0.0f, 1, 0, 0},
            {0.5f, 0.5f, 0.0f, 0, 1, 0},
            {-0.5f, 0.5f, 0.0f, 0, 0, 1},
        };

        VkBufferCreateInfo vbCI{};
        vbCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vbCI.size = sizeof(verts);
        vbCI.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vbCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &vbCI, nullptr, &vertexBuffer_));
        VkMemoryRequirements vmr;
        vkGetBufferMemoryRequirements(device_, vertexBuffer_, &vmr);
        vertexAlloc_ = allocator_.allocate(vmr.size,
                                           vmr.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                           vmr.alignment,
                                           "主顶点缓冲");
        VK_CHECK(vkBindBufferMemory(device_, vertexBuffer_, vertexAlloc_.memory, vertexAlloc_.offset));

        // 写入顶点数据
        void* mapped = (uint8_t*)vertexAlloc_.block->mappedPtr + vertexAlloc_.offset;
        std::memcpy(mapped, verts, sizeof(verts));

        std::cout << "✅ 资源使用 SimpleAllocator 分配完成\n";
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        VkClearValue clear{};
        clear.color.float32[0] = 0.02f;
        clear.color.float32[2] = 0.05f;
        clear.color.float32[3] = 1.0f;
        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass_;
        rp.framebuffer = framebuffers_[imageIndex];
        rp.renderArea = {{0, 0}, swapchainExtent_};
        rp.clearValueCount = 1;
        rp.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkViewport vp{0, 0, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &sc);
        VkBuffer vb[] = {vertexBuffer_};
        VkDeviceSize off[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t imgIdx = 0;
        VkResult r = vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX, imageAvailableSems_[currentFrame_], VK_NULL_HANDLE, &imgIdx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }
        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordCommandBuffer(commandBuffers_[currentFrame_], imgIdx);
        VkSemaphore ws[] = {imageAvailableSems_[currentFrame_]};
        VkPipelineStageFlags wst[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore ss[] = {renderFinishedSems_[currentFrame_]};
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = ws;
        si.pWaitDstStageMask = wst;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &commandBuffers_[currentFrame_];
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = ss;
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &si, inFlightFences_[currentFrame_]));
        VkSwapchainKHR scs[] = {swapchain_};
        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = ss;
        pi.swapchainCount = 1;
        pi.pSwapchains = scs;
        pi.pImageIndices = &imgIdx;
        r = vkQueuePresentKHR(presentQueue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        }
        currentFrame_ = (currentFrame_ + 1) % 2;
    }

    void mainLoop() {
        std::cout << "🎨 三角形渲染中（查看控制台的内存管理演示输出）...\n";
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    void createGraphicsPipeline() {
        VkShaderModule vert = createShaderModuleFromFile(device_, "uniform3d.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "triangle.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                   nullptr,
                                                   0,
                                                   VK_SHADER_STAGE_VERTEX_BIT,
                                                   vert,
                                                   "main",
                                                   nullptr},
                                                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                   nullptr,
                                                   0,
                                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                                                   frag,
                                                   "main",
                                                   nullptr}};
        VkVertexInputBindingDescription bind{0, sizeof(float) * 6, VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 2> attrs{
            {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12}}};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = 2;
        vi.pVertexAttributeDescriptions = attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                  nullptr,
                                                  0,
                                                  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                  VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.lineWidth = 1.0f;
        rs.cullMode = VK_CULL_MODE_NONE;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};
        dynS.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynS.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        dynS.pDynamicStates = dyn.data();
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &pipelineLayout_));
        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount = 2;
        pi.pStages = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vs;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pColorBlendState = &cb;
        pi.pDynamicState = &dynS;
        pi.layout = pipelineLayout_;
        pi.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }
    void createInstance() {
        VkApplicationInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.apiVersion = VK_API_VERSION_1_3;
        auto exts = getRequiredInstanceExtensions();
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &ai;
        ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
        ci.ppEnabledExtensionNames = exts.data();
        enablePortabilityBit(ci);

        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    }
    void createSurface() {
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
    }
    void pickPhysicalDevice() {
        uint32_t c = 0;
        vkEnumeratePhysicalDevices(instance_, &c, nullptr);
        std::vector<VkPhysicalDevice> devs(c);
        vkEnumeratePhysicalDevices(instance_, &c, devs.data());
        for (auto& d : devs)
            if (findQueueFamilies(d, surface_).isComplete() && checkDeviceExtensionSupport(d)) {
                physicalDevice_ = d;
                break;
            }
        if (!physicalDevice_)
            throw std::runtime_error("无合适GPU");
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(physicalDevice_, &p);
        std::cout << "✅ GPU: " << p.deviceName << "\n";
    }
    void createLogicalDevice() {
        queueIndices_ = findQueueFamilies(physicalDevice_, surface_);
        std::set<uint32_t> fams = {queueIndices_.graphicsFamily.value(), queueIndices_.presentFamily.value()};
        const float pri = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t f : fams) {
            VkDeviceQueueCreateInfo q{};
            q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueFamilyIndex = f;
            q.queueCount = 1;
            q.pQueuePriorities = &pri;
            qcis.push_back(q);
        }
        VkPhysicalDeviceFeatures feat{};
        feat.samplerAnisotropy = VK_TRUE;
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos = qcis.data();
        ci.pEnabledFeatures = &feat;
        ci.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_));
        vkGetDeviceQueue(device_, queueIndices_.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueIndices_.presentFamily.value(), 0, &presentQueue_);
    }
    void createSwapchain() {
        auto sc = querySwapChainSupport(physicalDevice_, surface_);
        auto fmt = chooseSwapSurfaceFormat(sc.formats);
        auto mode = chooseSwapPresentMode(sc.presentModes);
        swapchainExtent_ = chooseSwapExtent(sc.capabilities, window_);
        uint32_t n = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            n = std::min(n, sc.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_;
        ci.minImageCount = n;
        ci.imageFormat = fmt.format;
        ci.imageColorSpace = fmt.colorSpace;
        ci.imageExtent = swapchainExtent_;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = sc.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = mode;
        ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
        swapchainImages_.resize(n);
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, swapchainImages_.data());
        swapchainImageFormat_ = fmt.format;
    }
    void createImageViews() {
        swapchainImageViews_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image = swapchainImages_[i];
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = swapchainImageFormat_;
            ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY};
            ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]));
        }
    }
    void createRenderPass() {
        VkAttachmentDescription ca{};
        ca.format = swapchainImageFormat_;
        ca.samples = VK_SAMPLE_COUNT_1_BIT;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ca.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ca.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ca.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ca.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference cr{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &cr;
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = 1;
        rpi.pAttachments = &ca;
        rpi.subpassCount = 1;
        rpi.pSubpasses = &sp;
        rpi.dependencyCount = 1;
        rpi.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_));
    }
    void createFramebuffers() {
        framebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            VkImageView att[] = {swapchainImageViews_[i]};
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = renderPass_;
            ci.attachmentCount = 1;
            ci.pAttachments = att;
            ci.width = swapchainExtent_.width;
            ci.height = swapchainExtent_.height;
            ci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
        }
    }
    void createCommandPool() {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
    }
    void createCommandBuffers() {
        commandBuffers_.resize(2);
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 2;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()));
    }
    void createSyncObjects() {
        imageAvailableSems_.resize(2);
        renderFinishedSems_.resize(2);
        inFlightFences_.resize(2);
        VkSemaphoreCreateInfo sCI{};
        sCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fCI{};
        fCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < 2; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &sCI, nullptr, &imageAvailableSems_[i]));
            VK_CHECK(vkCreateSemaphore(device_, &sCI, nullptr, &renderFinishedSems_[i]));
            VK_CHECK(vkCreateFence(device_, &fCI, nullptr, &inFlightFences_[i]));
        }
    }
    void recreateSwapchain() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (!w || !h) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        for (auto& fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        createFramebuffers();
    }
    void cleanup() {
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        allocator_.free(vertexAlloc_);
        allocator_.printStats();
        allocator_.cleanup();
        for (int i = 0; i < 2; ++i) {
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (auto& fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
        std::cout << "✅ 清理完成。\n";
    }
};

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << " 第30章：Vulkan Memory Allocator (VMA) + 内存管理\n";
    std::cout << "\n";
    std::cout << " 核心问题：\n";
    std::cout << "   maxMemoryAllocationCount ≈ 4096（不能无限 vkAllocateMemory）\n";
    std::cout << "   解决：内存池（大块分配 + 次分配）\n";
    std::cout << "\n";
    std::cout << " 本章实现 SimpleAllocator 演示原理：\n";
    std::cout << "   - MemoryBlock：管理 64MB VkDeviceMemory\n";
    std::cout << "   - Free List：first-fit 次分配算法\n";
    std::cout << "   - 碎片合并：mergeFreeBlocks()\n";
    std::cout << "\n";
    std::cout << " 真实项目推荐：github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator\n";
    std::cout << "   vmaCreateAllocator / vmaCreateBuffer / vmaDestroyBuffer\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";
    Ch30App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
    return 0;
}
