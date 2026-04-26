/**
 * @file ch13_mipmaps.cpp
 * @brief 第13章：Mipmap 生成
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【什么是 Mipmap？】
 *
 *  Mipmap 是同一张纹理预先生成的一系列逐级缩小的版本：
 *
 *    Level 0: 原始纹理  256×256
 *    Level 1: 1/2 缩小  128×128
 *    Level 2: 1/4 缩小   64×64
 *    Level 3: 1/8 缩小   32×32
 *    ...
 *    Level N: 1×1
 *
 *  GPU 根据物体与相机的距离自动选择合适的 mip 层级：
 *    - 物体很远 → 使用小纹理（避免过采样锯齿）
 *    - 物体很近 → 使用原始纹理（保留细节）
 *
 * 【为什么需要 Mipmap？】
 *
 *  没有 Mipmap 时，远处物体每个像素覆盖大量纹素，GPU 跨越大范围采样，
 *  导致：
 *    - 闪烁（aliasing）：摄像机移动时不同帧采到不同纹素
 *    - 性能差：缓存效率低，读取纹素跨度大
 *
 *  Mipmap 以 1/3 额外内存开销解决上述问题，是几乎所有实时渲染的标准配置。
 *
 * 【Vulkan 中的 Mipmap 生成流程】
 *
 *  Vulkan 不提供自动生成 Mipmap 的 API，需要手动用
 *  vkCmdBlitImage 逐级下采样：
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  For each mip level i (from 1 to mipLevels-1):         │
 *  │                                                         │
 *  │  1. 将 level[i-1] 转换为 TRANSFER_SRC_OPTIMAL           │
 *  │  2. vkCmdBlitImage: level[i-1] → level[i]（缩小一半）   │
 *  │  3. 将 level[i-1] 转换为 SHADER_READ_ONLY_OPTIMAL       │
 *  │  4. 最后将 level[mipLevels-1] 转换为 SHADER_READ_ONLY   │
 *  └─────────────────────────────────────────────────────────┘
 *
 * 【修改的关键点】
 *
 *  1. 创建 VkImage 时设置 mipLevels > 1
 *  2. 创建 VkSampler 时配置 mipmapMode 和 maxLod
 *  3. 上传纹理后调用 generateMipmaps() 生成各级 mip
 *  4. 布局转换在 generateMipmaps 内完成（不再单独转换到 SHADER_READ_ONLY）
 *
 * 【vkCmdBlitImage vs vkCmdCopyImage】
 *
 *  vkCmdCopyImage   → 1:1 像素拷贝，不做缩放
 *  vkCmdBlitImage   → 支持缩放、格式转换，可用双线性过滤（BLIT 操作）
 *                     注意：某些格式不支持 BLIT（需检查 formatProperties）
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan_tutorial/utils.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int      MAX_FRAMES = 2;

// ─── 数据结构 ──────────────────────────────────────────────────────────────────

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription d{};
        d.binding = 0; d.stride = sizeof(Vertex); d.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return d;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 3> a{};
        a[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
        a[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
        a[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, texCoord)};
        return a;
    }
};

// 平面上的四边形，用于显示纹理
static const std::vector<Vertex> VERTICES = {
    {{-0.7f, -0.7f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
    {{ 0.7f, -0.7f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
    {{ 0.7f,  0.7f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
    {{-0.7f,  0.7f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
};
static const std::vector<uint16_t> INDICES = {0, 1, 2, 2, 3, 0};

// ─── SPIR-V（带 UBO + 纹理采样的着色器）──────────────────────────────────────
// 顶点着色器：UBO MVP 变换 + UV 坐标传递

// ─── 程序化棋盘格纹理生成 ─────────────────────────────────────────────────────

/// 生成高对比度棋盘格，在不同 mip 层级之间切换时视觉效果明显
static std::vector<uint8_t> generateCheckerboard(uint32_t width, uint32_t height)
{
    std::vector<uint8_t> pixels(width * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            // 每 16 像素切换颜色，使 mip 层级之间差异明显
            const bool isWhite = ((x / 16) + (y / 16)) % 2 == 0;
            const size_t idx = (y * width + x) * 4;
            pixels[idx + 0] = isWhite ? 255 : 30;   // R
            pixels[idx + 1] = isWhite ? 255 : 144;  // G （绿色版本）
            pixels[idx + 2] = isWhite ? 255 : 30;   // B
            pixels[idx + 3] = 255;                   // A
        }
    }
    return pixels;
}

// ─── 应用程序 ─────────────────────────────────────────────────────────────────

class Ch13App {
public:
    void run() { initWindow(); initVulkan(); mainLoop(); cleanup(); }

private:
    GLFWwindow*           window_              = nullptr;
    VkInstance            instance_            = VK_NULL_HANDLE;
    VkSurfaceKHR          surface_             = VK_NULL_HANDLE;
    VkPhysicalDevice      physicalDevice_      = VK_NULL_HANDLE;
    VkDevice              device_              = VK_NULL_HANDLE;
    VkQueue               graphicsQueue_       = VK_NULL_HANDLE;
    VkQueue               presentQueue_        = VK_NULL_HANDLE;
    VkSwapchainKHR        swapchain_           = VK_NULL_HANDLE;
    VkRenderPass          renderPass_          = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout_      = VK_NULL_HANDLE;
    VkPipeline            pipeline_            = VK_NULL_HANDLE;
    VkCommandPool         commandPool_         = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool_      = VK_NULL_HANDLE;

    // ─── 纹理（带 Mipmap） ────────────────────────────────────────────────────
    uint32_t       mipLevels_          = 1;    ///< Mipmap 层级数
    VkImage        textureImage_       = VK_NULL_HANDLE;
    VkDeviceMemory textureImageMemory_ = VK_NULL_HANDLE;
    VkImageView    textureImageView_   = VK_NULL_HANDLE;
    VkSampler      textureSampler_     = VK_NULL_HANDLE;

    // ─── 深度缓冲 ─────────────────────────────────────────────────────────────
    VkImage        depthImage_       = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
    VkImageView    depthImageView_   = VK_NULL_HANDLE;
    VkFormat       depthFormat_      = VK_FORMAT_UNDEFINED;

    std::vector<VkDescriptorSet>  descriptorSets_;
    std::vector<VkBuffer>         uniformBuffers_;
    std::vector<VkDeviceMemory>   uniformBuffersMemory_;
    std::vector<void*>            uniformBuffersMapped_;
    std::vector<VkImage>          swapchainImages_;
    std::vector<VkImageView>      swapchainImageViews_;
    std::vector<VkFramebuffer>    framebuffers_;
    std::vector<VkCommandBuffer>  commandBuffers_;
    VkFormat                      swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D                    swapchainExtent_{};
    QueueFamilyIndices            queueIndices_;
    VkBuffer       vertexBuffer_       = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory_ = VK_NULL_HANDLE;
    VkBuffer       indexBuffer_        = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory_  = VK_NULL_HANDLE;
    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence>     inFlightFences_;
    uint32_t                 currentFrame_ = 0;
    bool                     resized_      = false;

    void initWindow()
    {
        glfwInit(); glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch13 - Mipmap 生成", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_,
            [](GLFWwindow* w, int, int) {
                reinterpret_cast<Ch13App*>(glfwGetWindowUserPointer(w))->resized_ = true;
            });
    }

    void initVulkan()
    {
        createInstance(); createSurface(); pickPhysicalDevice(); createLogicalDevice();
        createSwapchain(); createImageViews();
        depthFormat_ = findDepthFormat();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createDepthResources();
        createFramebuffers(); createCommandPool();
        createTextureImage();      // ← 含 Mipmap 生成
        createTextureImageView();  // ← 含所有 mip 层
        createTextureSampler();    // ← 配置 Mipmap 采样
        createVertexBuffer(); createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool(); createDescriptorSets();
        createCommandBuffers(); createSyncObjects();
        std::cout << "\n✅ Mipmap 示例初始化完成！\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心新增：Mipmap 相关函数
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief 计算纹理需要的 Mipmap 层级数
     *
     * 公式：floor(log2(max(width, height))) + 1
     *   +1 是 level 0（原始图像）本身
     *
     * 例：256×256 → floor(log2(256))+1 = floor(8)+1 = 9 层
     *     (256→128→64→32→16→8→4→2→1)
     */
    static uint32_t calculateMipLevels(uint32_t width, uint32_t height)
    {
        return static_cast<uint32_t>(
            std::floor(std::log2(std::max(width, height)))) + 1;
    }

    void createTextureImage()
    {
        constexpr uint32_t TEX_W = 512;
        constexpr uint32_t TEX_H = 512;
        auto pixels = generateCheckerboard(TEX_W, TEX_H);
        VkDeviceSize imageSize = TEX_W * TEX_H * 4;

        // ① 计算 Mip 层级数
        mipLevels_ = calculateMipLevels(TEX_W, TEX_H);
        std::cout << "📐 纹理大小：" << TEX_W << "×" << TEX_H
                  << "  Mip 层级数：" << mipLevels_ << "\n";

        // ② 上传像素到 Staging Buffer
        VkBuffer       stagingBuf; VkDeviceMemory stagingMem;
        createBuffer(imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuf, stagingMem);

        void* data = nullptr;
        vkMapMemory(device_, stagingMem, 0, imageSize, 0, &data);
        std::memcpy(data, pixels.data(), static_cast<size_t>(imageSize));
        vkUnmapMemory(device_, stagingMem);

        // ③ 创建 VkImage，注意 mipLevels 和 usage 的变化
        createImage(TEX_W, TEX_H,
            mipLevels_,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_TILING_OPTIMAL,
            // TRANSFER_SRC_BIT：Blit 操作的源（用于生成 Mipmap）
            // TRANSFER_DST_BIT：上传原始像素的目标
            // SAMPLED_BIT：着色器采样
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            textureImage_, textureImageMemory_);

        // ④ 布局转换（仅 level 0 准备接收像素数据）
        transitionImageLayout(textureImage_,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            mipLevels_);

        // ⑤ 拷贝像素到 level 0
        copyBufferToImage(stagingBuf, textureImage_, TEX_W, TEX_H);

        // ⑥ 生成 Mipmap（同时完成所有层级的布局转换为 SHADER_READ_ONLY）
        generateMipmaps(textureImage_, VK_FORMAT_R8G8B8A8_SRGB, TEX_W, TEX_H, mipLevels_);

        vkDestroyBuffer(device_, stagingBuf, nullptr);
        vkFreeMemory(device_, stagingMem, nullptr);

        std::cout << "✅ 纹理图像已创建（含 " << mipLevels_ << " 个 Mip 层级）\n";
    }

    /**
     * @brief 使用 vkCmdBlitImage 逐级生成 Mipmap
     *
     * 每次 Blit 将上一级缩小一半，同时进行线性过滤（BLIT_LINEAR）
     * 生成完成后，所有层级均为 SHADER_READ_ONLY_OPTIMAL 布局。
     *
     * @param image      目标图像（已包含 level 0 的像素数据）
     * @param format     图像像素格式
     * @param texWidth   level 0 宽度
     * @param texHeight  level 0 高度
     * @param mipLevels  总层级数（含 level 0）
     */
    void generateMipmaps(VkImage image, VkFormat format,
                         int32_t texWidth, int32_t texHeight,
                         uint32_t mipLevels)
    {
        // 首先检查该格式是否支持线性 Blit（并非所有格式都支持）
        VkFormatProperties formatProps;
        vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &formatProps);
        if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
            throw std::runtime_error("纹理图像格式不支持线性 Blit！");

        VkCommandBuffer cmd = beginSingleTimeCommands();

        // 复用同一个 barrier，只修改不同的字段
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image               = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.subresourceRange.levelCount     = 1;   // 每次只操作一个层级

        int32_t mipWidth  = texWidth;
        int32_t mipHeight = texHeight;

        for (uint32_t i = 1; i < mipLevels; ++i) {
            // ── Step A: level[i-1] TRANSFER_DST → TRANSFER_SRC ───────────────
            // 说明：level[i-1] 刚被写入（来自 copyBufferToImage 或上一轮 Blit）
            //       现在要把它作为 Blit 的源
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            // ── Step B: Blit level[i-1] → level[i]（缩小一半）────────────────
            VkImageBlit blit{};
            // 源：level[i-1] 全图
            blit.srcOffsets[0]                = {0, 0, 0};
            blit.srcOffsets[1]                = {mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask    = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel      = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount    = 1;

            // 目标：level[i]，宽高各缩小一半（最小为 1）
            blit.dstOffsets[0]                = {0, 0, 0};
            blit.dstOffsets[1]                = {
                mipWidth  > 1 ? mipWidth  / 2 : 1,
                mipHeight > 1 ? mipHeight / 2 : 1,
                1
            };
            blit.dstSubresource.aspectMask    = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel      = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount    = 1;

            // VK_FILTER_LINEAR：双线性过滤缩小，质量好
            // VK_FILTER_NEAREST：最近邻，速度快但质量低
            vkCmdBlitImage(cmd,
                image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit,
                VK_FILTER_LINEAR);

            // ── Step C: level[i-1] TRANSFER_SRC → SHADER_READ_ONLY ───────────
            // 该层级已经用完，转换为最终可采样状态
            barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            // 更新下一轮的尺寸（除以2，最小为1）
            if (mipWidth  > 1) mipWidth  /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        // ── Step D: 最后一级 (mipLevels-1) 的布局转换 ─────────────────────────
        // 循环中未处理最后一级（它只是 Blit 目标，从未成为 Blit 源）
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        endSingleTimeCommands(cmd);
        std::cout << "✅ Mipmap 已生成（" << mipLevels << " 级）\n";
    }

    void createTextureImageView()
    {
        VkImageViewCreateInfo ci{};
        ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image    = textureImage_;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format   = VK_FORMAT_R8G8B8A8_SRGB;
        ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        ci.subresourceRange.baseMipLevel   = 0;
        ci.subresourceRange.levelCount     = mipLevels_;  // ← 包含所有 mip 层
        ci.subresourceRange.baseArrayLayer = 0;
        ci.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &textureImageView_));
    }

    void createTextureSampler()
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);

        VkSamplerCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter    = VK_FILTER_LINEAR;
        ci.minFilter    = VK_FILTER_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.anisotropyEnable = VK_TRUE;
        ci.maxAnisotropy    = props.limits.maxSamplerAnisotropy;
        ci.borderColor      = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        ci.unnormalizedCoordinates = VK_FALSE;
        ci.compareEnable    = VK_FALSE;

        // ─── Mipmap 采样配置（核心！） ────────────────────────────────────────
        // mipmapMode：mip 层级之间的插值方式
        //   NEAREST → 直接选最近的 mip 层（边界明显）
        //   LINEAR  → 在相邻 mip 层之间线性插值（更平滑，称为 trilinear filtering）
        ci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.mipLodBias   = 0.0f;   // LOD 偏移（负值强制使用更高分辨率 mip，正值相反）
        ci.minLod       = 0.0f;   // 允许使用的最小 mip 层级
        // maxLod：允许使用的最大 mip 层级（设为全部层级数量）
        // 如果设为 0，GPU 始终使用 level 0，等于关闭 Mipmap
        ci.maxLod       = static_cast<float>(mipLevels_);

        VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &textureSampler_));
        std::cout << "✅ 采样器已创建（Trilinear + 各向异性 x"
                  << props.limits.maxSamplerAnisotropy << "）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 辅助函数（布局转换修改版：支持 mipLevels 参数）
    // ═══════════════════════════════════════════════════════════════════════

    void transitionImageLayout(VkImage image, VkFormat /*fmt*/,
                                VkImageLayout oldLayout, VkImageLayout newLayout,
                                uint32_t mipLevels = 1)
    {
        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = oldLayout;
        barrier.newLayout           = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = image;
        barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1};

        VkPipelineStageFlags src = 0, dst = 0;
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            src = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dst = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            throw std::invalid_argument("不支持的布局转换");
        }
        vkCmdPipelineBarrier(cmd, src, dst, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        endSingleTimeCommands(cmd);
    }

    void copyBufferToImage(VkBuffer buf, VkImage img, uint32_t w, uint32_t h)
    {
        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent      = {w, h, 1};
        vkCmdCopyBufferToImage(cmd, buf, img,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        endSingleTimeCommands(cmd);
    }

    // ─── 渲染循环、初始化（复用前几章代码） ──────────────────────────────────

    void updateUniformBuffer(uint32_t frame)
    {
        static auto start = std::chrono::high_resolution_clock::now();
        float t = std::chrono::duration<float>(
            std::chrono::high_resolution_clock::now() - start).count();
        UniformBufferObject ubo{};
        ubo.model      = glm::rotate(glm::mat4(1.0f),
            t * glm::radians(20.0f), glm::vec3(0, 0, 1));
        ubo.view       = glm::lookAt(
            glm::vec3(2.0f, 2.0f, 2.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.projection = glm::perspective(glm::radians(45.0f),
            (float)swapchainExtent_.width / swapchainExtent_.height, 0.1f, 10.0f);
        ubo.projection[1][1] *= -1;
        std::memcpy(uniformBuffersMapped_[frame], &ubo, sizeof(ubo));
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t idx)
    {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        std::array<VkClearValue, 2> clears{};
        clears[0].color.float32[0] = 0.02f;
        clears[0].color.float32[1] = 0.02f;
        clears[0].color.float32[2] = 0.05f;
        clears[0].color.float32[3] = 1.0f;
        clears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp{};
        rp.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass      = renderPass_;
        rp.framebuffer     = framebuffers_[idx];
        rp.renderArea      = {{0, 0}, swapchainExtent_};
        rp.clearValueCount = static_cast<uint32_t>(clears.size());
        rp.pClearValues    = clears.data();

        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        VkViewport vp{0, 0, (float)swapchainExtent_.width,
                      (float)swapchainExtent_.height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &sc);

        VkBuffer vb[] = {vertexBuffer_};
        VkDeviceSize off[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
        vkCmdBindIndexBuffer(cmd, indexBuffer_, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout_, 0, 1, &descriptorSets_[currentFrame_], 0, nullptr);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(INDICES.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame()
    {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t imgIdx = 0;
        VkResult r = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
            imageAvailableSems_[currentFrame_], VK_NULL_HANDLE, &imgIdx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }
        updateUniformBuffer(currentFrame_);
        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordCommandBuffer(commandBuffers_[currentFrame_], imgIdx);

        VkSemaphore ws[] = {imageAvailableSems_[currentFrame_]};
        VkPipelineStageFlags wst[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore ss[] = {renderFinishedSems_[currentFrame_]};
        VkSubmitInfo si{};
        si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount   = 1; si.pWaitSemaphores   = ws;
        si.pWaitDstStageMask    = wst;
        si.commandBufferCount   = 1; si.pCommandBuffers   = &commandBuffers_[currentFrame_];
        si.signalSemaphoreCount = 1; si.pSignalSemaphores = ss;
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &si, inFlightFences_[currentFrame_]));

        VkSwapchainKHR sc2[] = {swapchain_};
        VkPresentInfoKHR pi{};
        pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = ss;
        pi.swapchainCount     = 1; pi.pSwapchains     = sc2;
        pi.pImageIndices      = &imgIdx;
        r = vkQueuePresentKHR(presentQueue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false; recreateSwapchain();
        }
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void mainLoop()
    {
        std::cout << "🎨 Mipmap 棋盘格纹理旋转中（ESC 退出）...\n";
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    // ─── 辅助函数 ─────────────────────────────────────────────────────────────

    uint32_t findMemoryType(uint32_t f, VkMemoryPropertyFlags p)
    {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((f & (1u << i)) && (mp.memoryTypes[i].propertyFlags & p) == p)
                return i;
        throw std::runtime_error("找不到合适内存类型");
    }

    void createBuffer(VkDeviceSize sz, VkBufferUsageFlags u,
                      VkMemoryPropertyFlags p, VkBuffer& b, VkDeviceMemory& m)
    {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = sz; ci.usage = u; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &ci, nullptr, &b));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device_, b, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size; ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, p);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &m));
        VK_CHECK(vkBindBufferMemory(device_, b, m, 0));
    }

    void createImage(uint32_t w, uint32_t h, uint32_t mips,
                     VkFormat fmt, VkImageTiling til,
                     VkImageUsageFlags u, VkMemoryPropertyFlags p,
                     VkImage& img, VkDeviceMemory& mem)
    {
        VkImageCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType     = VK_IMAGE_TYPE_2D;
        ci.extent        = {w, h, 1};
        ci.mipLevels     = mips;    // ← 关键：设置 Mip 层级数
        ci.arrayLayers   = 1;
        ci.format        = fmt; ci.tiling = til;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage         = u;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateImage(device_, &ci, nullptr, &img));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(device_, img, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size; ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, p);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &mem));
        VK_CHECK(vkBindImageMemory(device_, img, mem, 0));
    }

    VkCommandBuffer beginSingleTimeCommands()
    {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandPool = commandPool_; ai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device_, &ai, &cmd);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi); return cmd;
    }

    void endSingleTimeCommands(VkCommandBuffer cmd)
    {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue_);
        vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
    }

    void copyBuffer(VkBuffer s, VkBuffer d, VkDeviceSize sz)
    {
        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkBufferCopy r{}; r.size = sz;
        vkCmdCopyBuffer(cmd, s, d, 1, &r);
        endSingleTimeCommands(cmd);
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,
                                  VkImageTiling tiling, VkFormatFeatureFlags features)
    {
        for (VkFormat f : candidates) {
            VkFormatProperties p;
            vkGetPhysicalDeviceFormatProperties(physicalDevice_, f, &p);
            if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                (p.optimalTilingFeatures & features) == features) return f;
        }
        throw std::runtime_error("找不到支持格式");
    }

    VkFormat findDepthFormat()
    {
        return findSupportedFormat(
            {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
            VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    // ─── 常规 Vulkan 初始化（复用前几章，省略重复注释） ──────────────────────

    void createInstance()
    {
        VkApplicationInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.apiVersion = VK_API_VERSION_1_3;
        auto exts = getRequiredInstanceExtensions();
        VkInstanceCreateInfo ci{}; ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &ai;
        ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
        ci.ppEnabledExtensionNames = exts.data();
        ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    }
    void createSurface() {
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
    }
    void pickPhysicalDevice()
    {
        uint32_t c = 0; vkEnumeratePhysicalDevices(instance_, &c, nullptr);
        std::vector<VkPhysicalDevice> devs(c);
        vkEnumeratePhysicalDevices(instance_, &c, devs.data());
        for (auto& d : devs)
            if (findQueueFamilies(d, surface_).isComplete() && checkDeviceExtensionSupport(d)) {
                physicalDevice_ = d; break;
            }
        if (!physicalDevice_) throw std::runtime_error("无合适GPU");
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(physicalDevice_, &p);
        std::cout << "✅ GPU: " << p.deviceName << "\n";
    }
    void createLogicalDevice()
    {
        queueIndices_ = findQueueFamilies(physicalDevice_, surface_);
        std::set<uint32_t> fams = {queueIndices_.graphicsFamily.value(),
                                   queueIndices_.presentFamily.value()};
        const float pri = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t f : fams) {
            VkDeviceQueueCreateInfo q{};
            q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueFamilyIndex = f; q.queueCount = 1; q.pQueuePriorities = &pri;
            qcis.push_back(q);
        }
        VkPhysicalDeviceFeatures feat{}; feat.samplerAnisotropy = VK_TRUE;
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos = qcis.data(); ci.pEnabledFeatures = &feat;
        ci.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_));
        vkGetDeviceQueue(device_, queueIndices_.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueIndices_.presentFamily.value(),  0, &presentQueue_);
    }
    void createSwapchain()
    {
        auto sc = querySwapChainSupport(physicalDevice_, surface_);
        auto fmt = chooseSwapSurfaceFormat(sc.formats);
        auto mode = chooseSwapPresentMode(sc.presentModes);
        swapchainExtent_ = chooseSwapExtent(sc.capabilities, window_);
        uint32_t n = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            n = std::min(n, sc.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_; ci.minImageCount = n;
        ci.imageFormat = fmt.format; ci.imageColorSpace = fmt.colorSpace;
        ci.imageExtent = swapchainExtent_; ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = sc.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = mode; ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
        swapchainImages_.resize(n);
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, swapchainImages_.data());
        swapchainImageFormat_ = fmt.format;
    }
    void createImageViews()
    {
        swapchainImageViews_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image = swapchainImages_[i]; ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = swapchainImageFormat_;
            ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
            ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]));
        }
    }
    void createRenderPass()
    {
        VkAttachmentDescription ca{};
        ca.format = swapchainImageFormat_; ca.samples = VK_SAMPLE_COUNT_1_BIT;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ca.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ca.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ca.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ca.finalLayout   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription da{};
        da.format = depthFormat_; da.samples = VK_SAMPLE_COUNT_1_BIT;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        da.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        da.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        da.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        da.finalLayout   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference cr{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference dr{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1; sp.pColorAttachments = &cr;
        sp.pDepthStencilAttachment = &dr;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = {ca, da};
        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = static_cast<uint32_t>(attachments.size());
        rpi.pAttachments = attachments.data();
        rpi.subpassCount = 1; rpi.pSubpasses = &sp;
        rpi.dependencyCount = 1; rpi.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_));
    }
    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding ub{};
        ub.binding = 0; ub.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ub.descriptorCount = 1; ub.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding sb{};
        sb.binding = 1; sb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sb.descriptorCount = 1; sb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {ub, sb};
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = static_cast<uint32_t>(bindings.size());
        ci.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descriptorSetLayout_));
    }
    VkShaderModule createShaderModule(const uint32_t* c, size_t s)
    {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = s; ci.pCode = c;
        VkShaderModule m = VK_NULL_HANDLE;
        VK_CHECK(vkCreateShaderModule(device_, &ci, nullptr, &m)); return m;
    }
    void createGraphicsPipeline()
    {
        // texture.vert: vec3 pos + vec3 color + vec2 uv + UBO; texture.frag: texture sampler
        VkShaderModule vert = createShaderModuleFromFile(device_, "texture.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "texture.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT,   vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", nullptr};
        auto bd = Vertex::getBindingDescription();
        auto ad = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &bd;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(ad.size());
        vi.pVertexAttributeDescriptions = ad.data();
        VkPipelineInputAssemblyStateCreateInfo ia{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.lineWidth = 1.0f;
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp  = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &cba;
        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};
        dynS.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynS.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        dynS.pDynamicStates = dyn.data();
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1; pli.pSetLayouts = &descriptorSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &pipelineLayout_));
        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount = 2; pi.pStages = stages;
        pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vs; pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &cb; pi.pDynamicState = &dynS;
        pi.layout = pipelineLayout_; pi.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }
    void createDepthResources()
    {
        createImage(swapchainExtent_.width, swapchainExtent_.height, 1,
            depthFormat_, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            depthImage_, depthImageMemory_);
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = depthImage_; ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = depthFormat_;
        ci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &depthImageView_));
    }
    void createFramebuffers()
    {
        framebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            std::array<VkImageView, 2> att = {swapchainImageViews_[i], depthImageView_};
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = renderPass_;
            ci.attachmentCount = static_cast<uint32_t>(att.size());
            ci.pAttachments = att.data();
            ci.width = swapchainExtent_.width; ci.height = swapchainExtent_.height;
            ci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
        }
    }
    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
    }
    void createVertexBuffer()
    {
        VkDeviceSize sz = sizeof(VERTICES[0]) * VERTICES.size();
        VkBuffer sb; VkDeviceMemory sm;
        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sb, sm);
        void* d = nullptr; vkMapMemory(device_, sm, 0, sz, 0, &d);
        std::memcpy(d, VERTICES.data(), (size_t)sz); vkUnmapMemory(device_, sm);
        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer_, vertexBufferMemory_);
        copyBuffer(sb, vertexBuffer_, sz);
        vkDestroyBuffer(device_, sb, nullptr); vkFreeMemory(device_, sm, nullptr);
    }
    void createIndexBuffer()
    {
        VkDeviceSize sz = sizeof(INDICES[0]) * INDICES.size();
        VkBuffer sb; VkDeviceMemory sm;
        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sb, sm);
        void* d = nullptr; vkMapMemory(device_, sm, 0, sz, 0, &d);
        std::memcpy(d, INDICES.data(), (size_t)sz); vkUnmapMemory(device_, sm);
        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer_, indexBufferMemory_);
        copyBuffer(sb, indexBuffer_, sz);
        vkDestroyBuffer(device_, sb, nullptr); vkFreeMemory(device_, sm, nullptr);
    }
    void createUniformBuffers()
    {
        VkDeviceSize sz = sizeof(UniformBufferObject);
        uniformBuffers_.resize(MAX_FRAMES); uniformBuffersMemory_.resize(MAX_FRAMES);
        uniformBuffersMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(sz, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                uniformBuffers_[i], uniformBuffersMemory_[i]);
            vkMapMemory(device_, uniformBuffersMemory_[i], 0, sz, 0, &uniformBuffersMapped_[i]);
        }
    }
    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         static_cast<uint32_t>(MAX_FRAMES)};
        ps[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(MAX_FRAMES)};
        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = static_cast<uint32_t>(ps.size()); ci.pPoolSizes = ps.data();
        ci.maxSets = static_cast<uint32_t>(MAX_FRAMES);
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &descriptorPool_));
    }
    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> lays(MAX_FRAMES, descriptorSetLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES);
        ai.pSetLayouts = lays.data();
        descriptorSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, descriptorSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = uniformBuffers_[i]; bi.offset = 0;
            bi.range = sizeof(UniformBufferObject);
            VkDescriptorImageInfo ii{};
            ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ii.imageView   = textureImageView_;
            ii.sampler     = textureSampler_;
            std::array<VkWriteDescriptorSet, 2> ws{};
            ws[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                     descriptorSets_[i], 0, 0, 1,
                     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bi, nullptr};
            ws[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                     descriptorSets_[i], 1, 0, 1,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii, nullptr, nullptr};
            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(ws.size()), ws.data(), 0, nullptr);
        }
    }
    void createCommandBuffers()
    {
        commandBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()));
    }
    void createSyncObjects()
    {
        imageAvailableSems_.resize(MAX_FRAMES);
        renderFinishedSems_.resize(MAX_FRAMES);
        inFlightFences_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo sCI{}; sCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fCI{}; fCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &sCI, nullptr, &imageAvailableSems_[i]));
            VK_CHECK(vkCreateSemaphore(device_, &sCI, nullptr, &renderFinishedSems_[i]));
            VK_CHECK(vkCreateFence(device_, &fCI, nullptr, &inFlightFences_[i]));
        }
    }
    void recreateSwapchain()
    {
        int w = 0, h = 0; glfwGetFramebufferSize(window_, &w, &h);
        while (!w || !h) { glfwGetFramebufferSize(window_, &w, &h); glfwWaitEvents(); }
        vkDeviceWaitIdle(device_);
        for (auto& fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyImageView(device_, depthImageView_, nullptr);
        vkDestroyImage(device_, depthImage_, nullptr);
        vkFreeMemory(device_, depthImageMemory_, nullptr);
        for (auto& iv : swapchainImageViews_) vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain(); createImageViews();
        createDepthResources(); createFramebuffers();
    }
    void cleanup()
    {
        vkDestroySampler(device_, textureSampler_, nullptr);
        vkDestroyImageView(device_, textureImageView_, nullptr);
        vkDestroyImage(device_, textureImage_, nullptr);
        vkFreeMemory(device_, textureImageMemory_, nullptr);
        vkDestroyImageView(device_, depthImageView_, nullptr);
        vkDestroyImage(device_, depthImage_, nullptr);
        vkFreeMemory(device_, depthImageMemory_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, uniformBuffers_[i], nullptr);
            vkFreeMemory(device_, uniformBuffersMemory_[i], nullptr);
        }
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        vkDestroyBuffer(device_, indexBuffer_, nullptr);
        vkFreeMemory(device_, indexBufferMemory_, nullptr);
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexBufferMemory_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (auto& fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        for (auto& iv : swapchainImageViews_) vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_); glfwTerminate();
        std::cout << "✅ 清理完成。\n";
    }
};

int main()
{
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << " 第13章：Mipmap 生成\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
    Ch13App app;
    try { app.run(); } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n"; return 1;
    }
    return 0;
}
