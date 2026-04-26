/**
 * @file ch11_texture.cpp
 * @brief 第11章：纹理映射（Texture Mapping）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【纹理映射核心流程】
 *
 *  1. 加载图像文件（stb_image 库）
 *  2. 创建 Staging Buffer，上传像素数据
 *  3. 创建 VkImage（GPU 纹理对象）
 *  4. 执行布局转换（UNDEFINED → TRANSFER_DST_OPTIMAL）
 *  5. 拷贝像素数据到 VkImage
 *  6. 再次布局转换（TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL）
 *  7. 创建 VkImageView（描述如何访问纹理）
 *  8. 创建 VkSampler（描述如何采样：过滤、寻址模式等）
 *  9. 更新描述符集（binding=1 绑定采样器）
 * 10. 片段着色器中通过 texture() 采样
 *
 * 【VkSampler 关键参数】
 *
 *  ┌──────────────────────────────────────────────────────────┐
 *  │ 过滤模式（Filter）                                       │
 *  │   NEAREST → 最近邻（像素化效果）                         │
 *  │   LINEAR  → 线性插值（平滑）                             │
 *  │                                                          │
 *  │ 寻址模式（Address Mode）                                 │
 *  │   REPEAT          → 平铺重复（地板/砖墙）               │
 *  │   MIRRORED_REPEAT → 镜像重复                            │
 *  │   CLAMP_TO_EDGE   → 边缘拉伸                            │
 *  │   CLAMP_TO_BORDER → 超出部分显示边界颜色               │
 *  │                                                          │
 *  │ 各向异性过滤（Anisotropic Filtering）                   │
 *  │   斜视角下纹理的锐利度，maxAnisotropy=16 效果最好       │
 *  │                                                          │
 *  │ Mipmap                                                   │
 *  │   预生成的不同分辨率纹理，远处自动使用小纹理             │
 *  │   mipmapMode: NEAREST/LINEAR                             │
 *  └──────────────────────────────────────────────────────────┘
 *
 * 【图像布局转换（Image Layout Transition）】
 *
 *  通过 Pipeline Barrier（管线屏障）执行布局转换：
 *    - srcStageMask/dstStageMask：在哪个管线阶段前后插入屏障
 *    - srcAccessMask/dstAccessMask：等待/通知哪种内存访问
 *    - oldLayout/newLayout：布局转换方向
 *
 * 【依赖】
 *  本章需要 stb_image.h（单头文件图像加载库）
 *  请从 https://github.com/nothings/stb 获取并放到 include/ 目录
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
        d.binding=0;d.stride=sizeof(Vertex);d.inputRate=VK_VERTEX_INPUT_RATE_VERTEX;return d;
    }

    static std::array<VkVertexInputAttributeDescription,3> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription,3> a{};
        a[0]={0,0,VK_FORMAT_R32G32B32_SFLOAT,   offsetof(Vertex,pos)};
        a[1]={1,0,VK_FORMAT_R32G32B32_SFLOAT,   offsetof(Vertex,color)};
        a[2]={2,0,VK_FORMAT_R32G32_SFLOAT,      offsetof(Vertex,texCoord)};
        return a;
    }
};

static const std::vector<Vertex> VERTICES = {
    {{-0.5f,-0.5f, 0.0f},{1.0f,0.0f,0.0f},{1.0f,0.0f}},
    {{ 0.5f,-0.5f, 0.0f},{0.0f,1.0f,0.0f},{0.0f,0.0f}},
    {{ 0.5f, 0.5f, 0.0f},{0.0f,0.0f,1.0f},{0.0f,1.0f}},
    {{-0.5f, 0.5f, 0.0f},{1.0f,1.0f,1.0f},{1.0f,1.0f}},
};
static const std::vector<uint16_t> INDICES = {0,1,2,2,3,0};

/**
 * @brief 纹理管理类（演示核心 API，不依赖 stb_image）
 *
 * 此处使用程序生成的棋盘格纹理替代文件加载，
 * 实际项目中用 stb_image 加载 PNG/JPG 即可。
 */
class TextureManager {
public:
    static constexpr uint32_t TEX_WIDTH  = 256;
    static constexpr uint32_t TEX_HEIGHT = 256;

    /**
     * @brief 生成棋盘格纹理像素数据（RGBA8）
     */
    static std::vector<uint8_t> generateCheckerboard()
    {
        std::vector<uint8_t> pixels(TEX_WIDTH * TEX_HEIGHT * 4);
        for (uint32_t y = 0; y < TEX_HEIGHT; ++y) {
            for (uint32_t x = 0; x < TEX_WIDTH; ++x) {
                bool isWhite = ((x / 32) + (y / 32)) % 2 == 0;
                size_t idx = (y * TEX_WIDTH + x) * 4;
                pixels[idx+0] = isWhite ? 255 : 50;   // R
                pixels[idx+1] = isWhite ? 255 : 50;   // G
                pixels[idx+2] = isWhite ? 255 : 200;  // B
                pixels[idx+3] = 255;                   // A
            }
        }
        return pixels;
    }
};


class Ch11App {
public:
    void run() { initWindow(); initVulkan(); mainLoop(); cleanup(); }

private:
    GLFWwindow*            window_              = nullptr;
    VkInstance             instance_            = VK_NULL_HANDLE;
    VkSurfaceKHR           surface_             = VK_NULL_HANDLE;
    VkPhysicalDevice       physicalDevice_      = VK_NULL_HANDLE;
    VkDevice               device_              = VK_NULL_HANDLE;
    VkQueue                graphicsQueue_       = VK_NULL_HANDLE;
    VkQueue                presentQueue_        = VK_NULL_HANDLE;
    VkSwapchainKHR         swapchain_           = VK_NULL_HANDLE;
    VkRenderPass           renderPass_          = VK_NULL_HANDLE;
    VkDescriptorSetLayout  descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout       pipelineLayout_      = VK_NULL_HANDLE;
    VkPipeline             pipeline_            = VK_NULL_HANDLE;
    VkCommandPool          commandPool_         = VK_NULL_HANDLE;
    VkDescriptorPool       descriptorPool_      = VK_NULL_HANDLE;

    // ─── 纹理相关对象（新增） ──────────────────────────────────────────────
    VkImage        textureImage_       = VK_NULL_HANDLE;
    VkDeviceMemory textureImageMemory_ = VK_NULL_HANDLE;
    VkImageView    textureImageView_   = VK_NULL_HANDLE;
    VkSampler      textureSampler_     = VK_NULL_HANDLE;

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
        glfwInit();glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);
        window_=glfwCreateWindow(WIDTH,HEIGHT,"Ch11 - 纹理映射（棋盘格）",nullptr,nullptr);
        glfwSetWindowUserPointer(window_,this);
        glfwSetFramebufferSizeCallback(window_,[](GLFWwindow*w,int,int){reinterpret_cast<Ch11App*>(glfwGetWindowUserPointer(w))->resized_=true;});
    }

    void initVulkan()
    {
        createInstance();createSurface();pickPhysicalDevice();createLogicalDevice();
        createSwapchain();createImageViews();createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();createCommandPool();
        createTextureImage();       // ← 新增
        createTextureImageView();   // ← 新增
        createTextureSampler();     // ← 新增
        createVertexBuffer();createIndexBuffer();
        createUniformBuffers();createDescriptorPool();createDescriptorSets();
        createCommandBuffers();createSyncObjects();
        std::cout<<"\n✅ 纹理映射初始化完成！\n";
    }

    // ─── 核心新增：创建纹理图像 ───────────────────────────────────────────────

    void createTextureImage()
    {
        // 生成/加载像素数据（实际项目用 stb_image）
        auto pixels      = TextureManager::generateCheckerboard();
        uint32_t texW    = TextureManager::TEX_WIDTH;
        uint32_t texH    = TextureManager::TEX_HEIGHT;
        VkDeviceSize sz  = texW * texH * 4;

        // Step 1: 上传像素到 Staging Buffer
        VkBuffer       stagingBuf;
        VkDeviceMemory stagingMem;
        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuf, stagingMem);

        void* data = nullptr;
        vkMapMemory(device_, stagingMem, 0, sz, 0, &data);
        std::memcpy(data, pixels.data(), static_cast<size_t>(sz));
        vkUnmapMemory(device_, stagingMem);

        // Step 2: 创建 VkImage
        createImage(texW, texH,
            VK_FORMAT_R8G8B8A8_SRGB,           // 像素格式（SRGB 颜色空间）
            VK_IMAGE_TILING_OPTIMAL,            // GPU 最优存储布局（非线性，性能最好）
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |  // 接收拷贝
            VK_IMAGE_USAGE_SAMPLED_BIT,         // 在着色器中采样
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            textureImage_, textureImageMemory_);

        // Step 3: 布局转换 UNDEFINED → TRANSFER_DST_OPTIMAL
        transitionImageLayout(textureImage_,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        // Step 4: 将 Buffer 数据拷贝到 Image
        copyBufferToImage(stagingBuf, textureImage_, texW, texH);

        // Step 5: 布局转换 TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
        transitionImageLayout(textureImage_,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device_, stagingBuf, nullptr);
        vkFreeMemory(device_, stagingMem, nullptr);

        std::cout<<"✅ 纹理图像已创建（"<<texW<<"x"<<texH<<" 棋盘格）\n";
    }

    void createImage(uint32_t width, uint32_t height,
                     VkFormat format, VkImageTiling tiling,
                     VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
                     VkImage& image, VkDeviceMemory& imageMemory)
    {
        VkImageCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType     = VK_IMAGE_TYPE_2D;
        ci.extent        = {width, height, 1};   // depth=1 for 2D
        ci.mipLevels     = 1;
        ci.arrayLayers   = 1;
        ci.format        = format;
        ci.tiling        = tiling;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage         = usage;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateImage(device_, &ci, nullptr, &image));

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device_, image, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, properties);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &imageMemory));
        VK_CHECK(vkBindImageMemory(device_, image, imageMemory, 0));
    }

    // ─── 图像布局转换（通过 Pipeline Barrier） ────────────────────────────────

    void transitionImageLayout(VkImage image, VkFormat /*format*/,
                                VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        VkCommandBuffer cmd = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = oldLayout;
        barrier.newLayout           = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // 不转移队列族所有权
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = image;
        barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkPipelineStageFlags srcStage = 0;
        VkPipelineStageFlags dstStage = 0;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            // 无需等待任何操作，开始写入
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;   // 最早阶段
            dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            // 等待写操作完成，然后允许着色器读取
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            throw std::invalid_argument("不支持的布局转换");
        }

        vkCmdPipelineBarrier(
            cmd,
            srcStage, dstStage,
            0,           // dependencyFlags
            0, nullptr,  // memory barriers
            0, nullptr,  // buffer barriers
            1, &barrier  // image barriers
        );

        endSingleTimeCommands(cmd);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
    {
        VkCommandBuffer cmd = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0;   // 0 = 紧密排列
        region.bufferImageHeight = 0;
        region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset       = {0, 0, 0};
        region.imageExtent       = {width, height, 1};

        vkCmdCopyBufferToImage(cmd, buffer, image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleTimeCommands(cmd);
    }

    // ─── 核心新增：创建纹理 ImageView ────────────────────────────────────────

    void createTextureImageView()
    {
        VkImageViewCreateInfo ci{};
        ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image    = textureImage_;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format   = VK_FORMAT_R8G8B8A8_SRGB;
        ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &textureImageView_));
        std::cout<<"✅ 纹理 ImageView 已创建\n";
    }

    // ─── 核心新增：创建采样器 ────────────────────────────────────────────────

    void createTextureSampler()
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);

        VkSamplerCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        // 放大/缩小过滤：LINEAR = 线性插值（平滑）
        ci.magFilter    = VK_FILTER_LINEAR;
        ci.minFilter    = VK_FILTER_LINEAR;
        // 超出 [0,1] UV 范围时的处理：REPEAT = 平铺
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        // 各向异性过滤（使斜视角纹理更清晰）
        ci.anisotropyEnable = VK_TRUE;
        ci.maxAnisotropy    = props.limits.maxSamplerAnisotropy;   // 使用最大值（通常16）
        // 边界颜色（CLAMP_TO_BORDER 时使用）
        ci.borderColor      = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        // unnormalizedCoordinates：是否使用像素坐标（而非 [0,1]）
        ci.unnormalizedCoordinates = VK_FALSE;
        // compareEnable：阴影贴图中的深度比较采样
        ci.compareEnable    = VK_FALSE;
        // Mipmap 配置
        ci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        ci.mipLodBias   = 0.0f;
        ci.minLod       = 0.0f;
        ci.maxLod       = 0.0f;   // 只有1个 mip 层

        VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &textureSampler_));
        std::cout<<"✅ 采样器已创建（各向异性过滤 x"<<props.limits.maxSamplerAnisotropy<<"）\n";
    }

    void createDescriptorSetLayout()
    {
        // binding=0：UBO（顶点着色器）
        VkDescriptorSetLayoutBinding uboBinding{};
        uboBinding.binding=0;uboBinding.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.descriptorCount=1;uboBinding.stageFlags=VK_SHADER_STAGE_VERTEX_BIT;

        // binding=1：Combined Image Sampler（片段着色器）
        // Combined Image Sampler = VkImageView + VkSampler 合并为一个描述符
        VkDescriptorSetLayoutBinding samplerBinding{};
        samplerBinding.binding=1;
        samplerBinding.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerBinding.descriptorCount=1;
        samplerBinding.stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboBinding, samplerBinding};
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount=static_cast<uint32_t>(bindings.size());ci.pBindings=bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_,&ci,nullptr,&descriptorSetLayout_));
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0]={VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,        static_cast<uint32_t>(MAX_FRAMES)};
        poolSizes[1]={VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,static_cast<uint32_t>(MAX_FRAMES)};
        VkDescriptorPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount=static_cast<uint32_t>(poolSizes.size());ci.pPoolSizes=poolSizes.data();
        ci.maxSets=static_cast<uint32_t>(MAX_FRAMES);
        VK_CHECK(vkCreateDescriptorPool(device_,&ci,nullptr,&descriptorPool_));
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES,descriptorSetLayout_);
        VkDescriptorSetAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool=descriptorPool_;ai.descriptorSetCount=static_cast<uint32_t>(MAX_FRAMES);
        ai.pSetLayouts=layouts.data();
        descriptorSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_,&ai,descriptorSets_.data()));

        for(int i=0;i<MAX_FRAMES;++i){
            VkDescriptorBufferInfo bufInfo{};
            bufInfo.buffer=uniformBuffers_[i];bufInfo.offset=0;bufInfo.range=sizeof(UniformBufferObject);

            // ── 纹理描述符 ────────────────────────────────────────────────────
            VkDescriptorImageInfo imgInfo{};
            imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imgInfo.imageView   = textureImageView_;
            imgInfo.sampler     = textureSampler_;

            std::array<VkWriteDescriptorSet,2> writes{};
            writes[0].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet=descriptorSets_[i];writes[0].dstBinding=0;
            writes[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].descriptorCount=1;writes[0].pBufferInfo=&bufInfo;

            writes[1].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet=descriptorSets_[i];writes[1].dstBinding=1;
            writes[1].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].descriptorCount=1;writes[1].pImageInfo=&imgInfo;

            vkUpdateDescriptorSets(device_,static_cast<uint32_t>(writes.size()),writes.data(),0,nullptr);
        }
    }

    void updateUniformBuffer(uint32_t frame)
    {
        static auto start=std::chrono::high_resolution_clock::now();
        float t=std::chrono::duration<float>(std::chrono::high_resolution_clock::now()-start).count();
        UniformBufferObject ubo{};
        ubo.model=glm::rotate(glm::mat4(1.0f),t*glm::radians(90.0f),glm::vec3(0,0,1));
        ubo.view=glm::lookAt(glm::vec3(2,2,2),glm::vec3(0,0,0),glm::vec3(0,0,1));
        ubo.projection=glm::perspective(glm::radians(45.0f),(float)swapchainExtent_.width/swapchainExtent_.height,0.1f,10.0f);
        ubo.projection[1][1]*=-1;
        std::memcpy(uniformBuffersMapped_[frame],&ubo,sizeof(ubo));
    }

    void recordCommandBuffer(VkCommandBuffer cmd,uint32_t idx)
    {
        VkCommandBufferBeginInfo bi{};bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd,&bi));
        VkClearValue clear={{{0.02f,0.02f,0.05f,1.0f}}};
        VkRenderPassBeginInfo rpBI{};rpBI.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBI.renderPass=renderPass_;rpBI.framebuffer=framebuffers_[idx];
        rpBI.renderArea={{0,0},swapchainExtent_};rpBI.clearValueCount=1;rpBI.pClearValues=&clear;
        vkCmdBeginRenderPass(cmd,&rpBI,VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline_);
        VkViewport vp{0,0,(float)swapchainExtent_.width,(float)swapchainExtent_.height,0,1};
        vkCmdSetViewport(cmd,0,1,&vp);
        VkRect2D sc{{0,0},swapchainExtent_};vkCmdSetScissor(cmd,0,1,&sc);
        VkBuffer vb[]={vertexBuffer_};VkDeviceSize off[]={0};
        vkCmdBindVertexBuffers(cmd,0,1,vb,off);
        vkCmdBindIndexBuffer(cmd,indexBuffer_,0,VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipelineLayout_,0,1,&descriptorSets_[currentFrame_],0,nullptr);
        vkCmdDrawIndexed(cmd,static_cast<uint32_t>(INDICES.size()),1,0,0,0);
        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame()
    {
        vkWaitForFences(device_,1,&inFlightFences_[currentFrame_],VK_TRUE,UINT64_MAX);
        uint32_t imgIdx=0;
        VkResult r=vkAcquireNextImageKHR(device_,swapchain_,UINT64_MAX,imageAvailableSems_[currentFrame_],VK_NULL_HANDLE,&imgIdx);
        if(r==VK_ERROR_OUT_OF_DATE_KHR){recreateSwapchain();return;}
        updateUniformBuffer(currentFrame_);
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
        VkSwapchainKHR sc2[]={swapchain_};
        VkPresentInfoKHR pi{};pi.sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount=1;pi.pWaitSemaphores=ss;pi.swapchainCount=1;pi.pSwapchains=sc2;pi.pImageIndices=&imgIdx;
        r=vkQueuePresentKHR(presentQueue_,&pi);
        if(r==VK_ERROR_OUT_OF_DATE_KHR||r==VK_SUBOPTIMAL_KHR||resized_){resized_=false;recreateSwapchain();}
        currentFrame_=(currentFrame_+1)%MAX_FRAMES;
    }

    void mainLoop()
    {
        std::cout<<"🎨 棋盘格纹理旋转中（按 ESC 退出）...\n";
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
        for(uint32_t i=0;i<mp.memoryTypeCount;++i)
            if((f&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&p)==p)return i;
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

    VkCommandBuffer beginSingleTimeCommands()
    {
        VkCommandBufferAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandPool=commandPool_;ai.commandBufferCount=1;
        VkCommandBuffer cmd=VK_NULL_HANDLE;vkAllocateCommandBuffers(device_,&ai,&cmd);
        VkCommandBufferBeginInfo bi{};bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;vkBeginCommandBuffer(cmd,&bi);return cmd;
    }

    void endSingleTimeCommands(VkCommandBuffer cmd)
    {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;si.commandBufferCount=1;si.pCommandBuffers=&cmd;
        vkQueueSubmit(graphicsQueue_,1,&si,VK_NULL_HANDLE);vkQueueWaitIdle(graphicsQueue_);
        vkFreeCommandBuffers(device_,commandPool_,1,&cmd);
    }

    void copyBuffer(VkBuffer s,VkBuffer d,VkDeviceSize sz)
    {
        auto cmd=beginSingleTimeCommands();VkBufferCopy r{};r.size=sz;vkCmdCopyBuffer(cmd,s,d,1,&r);endSingleTimeCommands(cmd);
    }

    void createVertexBuffer()
    {
        VkDeviceSize sz=sizeof(VERTICES[0])*VERTICES.size();VkBuffer sb;VkDeviceMemory sm;
        createBuffer(sz,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,sb,sm);
        void*d=nullptr;vkMapMemory(device_,sm,0,sz,0,&d);std::memcpy(d,VERTICES.data(),(size_t)sz);vkUnmapMemory(device_,sm);
        createBuffer(sz,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,vertexBuffer_,vertexBufferMemory_);
        copyBuffer(sb,vertexBuffer_,sz);vkDestroyBuffer(device_,sb,nullptr);vkFreeMemory(device_,sm,nullptr);
    }

    void createIndexBuffer()
    {
        VkDeviceSize sz=sizeof(INDICES[0])*INDICES.size();VkBuffer sb;VkDeviceMemory sm;
        createBuffer(sz,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,sb,sm);
        void*d=nullptr;vkMapMemory(device_,sm,0,sz,0,&d);std::memcpy(d,INDICES.data(),(size_t)sz);vkUnmapMemory(device_,sm);
        createBuffer(sz,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,indexBuffer_,indexBufferMemory_);
        copyBuffer(sb,indexBuffer_,sz);vkDestroyBuffer(device_,sb,nullptr);vkFreeMemory(device_,sm,nullptr);
    }

    void createUniformBuffers()
    {
        VkDeviceSize sz=sizeof(UniformBufferObject);
        uniformBuffers_.resize(MAX_FRAMES);uniformBuffersMemory_.resize(MAX_FRAMES);uniformBuffersMapped_.resize(MAX_FRAMES);
        for(int i=0;i<MAX_FRAMES;++i){
            createBuffer(sz,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,uniformBuffers_[i],uniformBuffersMemory_[i]);
            vkMapMemory(device_,uniformBuffersMemory_[i],0,sz,0,&uniformBuffersMapped_[i]);
        }
    }

    void recreateSwapchain()
    {
        int w=0,h=0;glfwGetFramebufferSize(window_,&w,&h);
        while(!w||!h){glfwGetFramebufferSize(window_,&w,&h);glfwWaitEvents();}
        vkDeviceWaitIdle(device_);
        for(auto&fb:framebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);
        for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);
        vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        createSwapchain();createImageViews();createFramebuffers();
    }

    // ─── 其余初始化（复用前几章代码）────────────────────────────────────────


    void createInstance()
    {
        VkApplicationInfo ai{};ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;ai.apiVersion=VK_API_VERSION_1_3;
        auto exts=getRequiredInstanceExtensions();
        VkInstanceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ci.pApplicationInfo=&ai;
        ci.enabledExtensionCount=static_cast<uint32_t>(exts.size());ci.ppEnabledExtensionNames=exts.data();
        ci.flags|=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}
        VK_CHECK(vkCreateInstance(&ci,nullptr,&instance_));
    }
    void createSurface(){VK_CHECK(glfwCreateWindowSurface(instance_,window_,nullptr,&surface_));}
    void pickPhysicalDevice()
    {
        uint32_t c=0;vkEnumeratePhysicalDevices(instance_,&c,nullptr);
        std::vector<VkPhysicalDevice> devs(c);vkEnumeratePhysicalDevices(instance_,&c,devs.data());
        for(auto&d:devs)if(findQueueFamilies(d,surface_).isComplete()&&checkDeviceExtensionSupport(d)){physicalDevice_=d;break;}
        if(!physicalDevice_)throw std::runtime_error("无合适GPU");
    }
    void createLogicalDevice()
    {
        queueIndices_=findQueueFamilies(physicalDevice_,surface_);
        std::set<uint32_t> fams={queueIndices_.graphicsFamily.value(),queueIndices_.presentFamily.value()};
        const float pri=1.0f;std::vector<VkDeviceQueueCreateInfo> qcis;
        for(uint32_t f:fams){VkDeviceQueueCreateInfo q{};q.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;q.queueFamilyIndex=f;q.queueCount=1;q.pQueuePriorities=&pri;qcis.push_back(q);}
        VkPhysicalDeviceFeatures feat{};feat.samplerAnisotropy=VK_TRUE;
        VkDeviceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount=static_cast<uint32_t>(qcis.size());ci.pQueueCreateInfos=qcis.data();ci.pEnabledFeatures=&feat;
        ci.enabledExtensionCount=static_cast<uint32_t>(DEVICE_EXTENSIONS.size());ci.ppEnabledExtensionNames=DEVICE_EXTENSIONS.data();
        if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}
        VK_CHECK(vkCreateDevice(physicalDevice_,&ci,nullptr,&device_));
        vkGetDeviceQueue(device_,queueIndices_.graphicsFamily.value(),0,&graphicsQueue_);
        vkGetDeviceQueue(device_,queueIndices_.presentFamily.value(),0,&presentQueue_);
    }
    void createSwapchain()
    {
        auto sc=querySwapChainSupport(physicalDevice_,surface_);
        auto fmt=chooseSwapSurfaceFormat(sc.formats);auto mode=chooseSwapPresentMode(sc.presentModes);
        swapchainExtent_=chooseSwapExtent(sc.capabilities,window_);
        uint32_t n=sc.capabilities.minImageCount+1;
        if(sc.capabilities.maxImageCount>0)n=std::min(n,sc.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{};ci.sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface=surface_;ci.minImageCount=n;ci.imageFormat=fmt.format;ci.imageColorSpace=fmt.colorSpace;
        ci.imageExtent=swapchainExtent_;ci.imageArrayLayers=1;ci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE;ci.preTransform=sc.capabilities.currentTransform;
        ci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;ci.presentMode=mode;ci.clipped=VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_,&ci,nullptr,&swapchain_));
        vkGetSwapchainImagesKHR(device_,swapchain_,&n,nullptr);swapchainImages_.resize(n);
        vkGetSwapchainImagesKHR(device_,swapchain_,&n,swapchainImages_.data());swapchainImageFormat_=fmt.format;
    }
    void createImageViews()
    {
        swapchainImageViews_.resize(swapchainImages_.size());
        for(size_t i=0;i<swapchainImages_.size();++i){
            VkImageViewCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image=swapchainImages_[i];ci.viewType=VK_IMAGE_VIEW_TYPE_2D;ci.format=swapchainImageFormat_;
            ci.components={VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY};
            ci.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            VK_CHECK(vkCreateImageView(device_,&ci,nullptr,&swapchainImageViews_[i]));
        }
    }
    void createRenderPass()
    {
        VkAttachmentDescription ca{};ca.format=swapchainImageFormat_;ca.samples=VK_SAMPLE_COUNT_1_BIT;
        ca.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;ca.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
        ca.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;ca.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ca.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;ca.finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference cr{};cr.attachment=0;cr.layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkSubpassDescription sp{};sp.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sp.colorAttachmentCount=1;sp.pColorAttachments=&cr;
        VkSubpassDependency dep{};dep.srcSubpass=VK_SUBPASS_EXTERNAL;dep.dstSubpass=0;
        dep.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;dep.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask=0;dep.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo rpi{};rpi.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount=1;rpi.pAttachments=&ca;rpi.subpassCount=1;rpi.pSubpasses=&sp;rpi.dependencyCount=1;rpi.pDependencies=&dep;
        VK_CHECK(vkCreateRenderPass(device_,&rpi,nullptr,&renderPass_));
    }
    VkShaderModule createShaderModule(const uint32_t*c,size_t s)
    {
        VkShaderModuleCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;ci.codeSize=s;ci.pCode=c;
        VkShaderModule m=VK_NULL_HANDLE;VK_CHECK(vkCreateShaderModule(device_,&ci,nullptr,&m));return m;
    }
    void createGraphicsPipeline()
    {
        // texture.vert: vec3 pos + vec3 color + vec2 uv + UBO; texture.frag: texture sampler
        VkShaderModule vert = createShaderModuleFromFile(device_, "texture.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "texture.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vert,"main",nullptr};
        stages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,frag,"main",nullptr};
        auto bd=Vertex::getBindingDescription();auto ad=Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vi{};vi.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount=1;vi.pVertexBindingDescriptions=&bd;
        vi.vertexAttributeDescriptionCount=static_cast<uint32_t>(ad.size());vi.pVertexAttributeDescriptions=ad.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,nullptr,0,VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,nullptr,0,1,nullptr,1,nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};rs.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode=VK_POLYGON_MODE_FILL;rs.lineWidth=1.0f;rs.cullMode=VK_CULL_MODE_BACK_BIT;rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};ms.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};cba.colorWriteMask=0xf;
        VkPipelineColorBlendStateCreateInfo cb{};cb.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;cb.attachmentCount=1;cb.pAttachments=&cba;
        std::vector<VkDynamicState> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{};ds.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;ds.dynamicStateCount=static_cast<uint32_t>(dyn.size());ds.pDynamicStates=dyn.data();
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.setLayoutCount=1;pli.pSetLayouts=&descriptorSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_,&pli,nullptr,&pipelineLayout_));
        VkGraphicsPipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount=2;pi.pStages=stages;pi.pVertexInputState=&vi;pi.pInputAssemblyState=&ia;
        pi.pViewportState=&vs;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pColorBlendState=&cb;
        pi.pDynamicState=&ds;pi.layout=pipelineLayout_;pi.renderPass=renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&pipeline_));
        vkDestroyShaderModule(device_,frag,nullptr);vkDestroyShaderModule(device_,vert,nullptr);
    }
    void createFramebuffers()
    {
        framebuffers_.resize(swapchainImageViews_.size());
        for(size_t i=0;i<swapchainImageViews_.size();++i){
            VkImageView att[]={swapchainImageViews_[i]};
            VkFramebufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass=renderPass_;ci.attachmentCount=1;ci.pAttachments=att;
            ci.width=swapchainExtent_.width;ci.height=swapchainExtent_.height;ci.layers=1;
            VK_CHECK(vkCreateFramebuffer(device_,&ci,nullptr,&framebuffers_[i]));
        }
    }
    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;ci.queueFamilyIndex=queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_,&ci,nullptr,&commandPool_));
    }
    void createCommandBuffers()
    {
        commandBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool=commandPool_;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=static_cast<uint32_t>(commandBuffers_.size());
        VK_CHECK(vkAllocateCommandBuffers(device_,&ai,commandBuffers_.data()));
    }
    void createSyncObjects()
    {
        imageAvailableSems_.resize(MAX_FRAMES);renderFinishedSems_.resize(MAX_FRAMES);inFlightFences_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo sCI{};sCI.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fCI{};fCI.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;fCI.flags=VK_FENCE_CREATE_SIGNALED_BIT;
        for(int i=0;i<MAX_FRAMES;++i){
            VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&imageAvailableSems_[i]));
            VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&renderFinishedSems_[i]));
            VK_CHECK(vkCreateFence(device_,&fCI,nullptr,&inFlightFences_[i]));
        }
    }

    void cleanup()
    {
        vkDestroySampler(device_,textureSampler_,nullptr);
        vkDestroyImageView(device_,textureImageView_,nullptr);
        vkDestroyImage(device_,textureImage_,nullptr);
        vkFreeMemory(device_,textureImageMemory_,nullptr);
        for(int i=0;i<MAX_FRAMES;++i){vkDestroyBuffer(device_,uniformBuffers_[i],nullptr);vkFreeMemory(device_,uniformBuffersMemory_[i],nullptr);}
        vkDestroyDescriptorPool(device_,descriptorPool_,nullptr);
        vkDestroyDescriptorSetLayout(device_,descriptorSetLayout_,nullptr);
        vkDestroyBuffer(device_,indexBuffer_,nullptr);vkFreeMemory(device_,indexBufferMemory_,nullptr);
        vkDestroyBuffer(device_,vertexBuffer_,nullptr);vkFreeMemory(device_,vertexBufferMemory_,nullptr);
        for(int i=0;i<MAX_FRAMES;++i){vkDestroySemaphore(device_,imageAvailableSems_[i],nullptr);vkDestroySemaphore(device_,renderFinishedSems_[i],nullptr);vkDestroyFence(device_,inFlightFences_[i],nullptr);}
        vkDestroyCommandPool(device_,commandPool_,nullptr);
        for(auto&fb:framebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);
        vkDestroyPipeline(device_,pipeline_,nullptr);vkDestroyPipelineLayout(device_,pipelineLayout_,nullptr);
        vkDestroyRenderPass(device_,renderPass_,nullptr);
        for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);
        vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        vkDestroyDevice(device_,nullptr);vkDestroySurfaceKHR(instance_,surface_,nullptr);vkDestroyInstance(instance_,nullptr);
        glfwDestroyWindow(window_);glfwTerminate();
        std::cout<<"✅ 清理完成。\n";
    }
};

int main()
{
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << " 第11章：纹理映射（程序化棋盘格纹理 + 旋转）\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    Ch11App app;
    try { app.run(); } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n"; return 1;
    }
    return 0;
}
