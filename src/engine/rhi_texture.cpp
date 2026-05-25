/**
 * @file rhi_texture.cpp
 * @brief 第63章：纹理系统实现
 *
 * 实现内容：
 *   Texture        — 基础 VkImage + View + Sampler RAII
 *   Texture2D      — stb_image 文件加载 + Mip 生成
 *   TextureCube    — Cubemap 6 面
 *   RenderTarget   — 离屏渲染目标（Color / Depth）
 *   TextureCache   — 路径 → Texture2D 缓存
 *
 * 关键依赖：
 *   stb_image.h（已通过 stb_image_impl.cpp 定义 STB_IMAGE_IMPLEMENTATION）
 */

#include <vulkan_tutorial/engine/rhi_texture.hpp>
#include <vulkan_tutorial/engine/rhi_buffer.hpp>
#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>

#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine {

// ─── Texture：Move 语义 ──────────────────────────────────────────────────────

Texture::Texture(Texture&& o) noexcept
    : dev_(o.dev_), image_(o.image_), memory_(o.memory_),
      view_(o.view_), sampler_(o.sampler_), format_(o.format_),
      width_(o.width_), height_(o.height_), mipLevels_(o.mipLevels_),
      layers_(o.layers_), layout_(o.layout_)
{
    o.dev_     = nullptr;
    o.image_   = VK_NULL_HANDLE;
    o.memory_  = VK_NULL_HANDLE;
    o.view_    = VK_NULL_HANDLE;
    o.sampler_ = VK_NULL_HANDLE;
}

Texture& Texture::operator=(Texture&& o) noexcept
{
    if (this != &o) {
        destroy();
        dev_      = o.dev_;
        image_    = o.image_;
        memory_   = o.memory_;
        view_     = o.view_;
        sampler_  = o.sampler_;
        format_   = o.format_;
        width_    = o.width_;
        height_   = o.height_;
        mipLevels_ = o.mipLevels_;
        layers_   = o.layers_;
        layout_   = o.layout_;
        o.dev_    = nullptr;
        o.image_  = VK_NULL_HANDLE;
        o.memory_ = VK_NULL_HANDLE;
        o.view_   = VK_NULL_HANDLE;
        o.sampler_ = VK_NULL_HANDLE;
    }
    return *this;
}

// ─── Texture::destroy ───────────────────────────────────────────────────────

void Texture::destroy()
{
    if (!dev_ || image_ == VK_NULL_HANDLE) return;
    VkDevice d = dev_->device();
    if (sampler_ != VK_NULL_HANDLE) { vkDestroySampler(d, sampler_, nullptr); sampler_ = VK_NULL_HANDLE; }
    if (view_    != VK_NULL_HANDLE) { vkDestroyImageView(d, view_, nullptr);  view_    = VK_NULL_HANDLE; }
    if (image_   != VK_NULL_HANDLE) { vkDestroyImage(d, image_, nullptr);     image_   = VK_NULL_HANDLE; }
    if (memory_  != VK_NULL_HANDLE) { vkFreeMemory(d, memory_, nullptr);      memory_  = VK_NULL_HANDLE; }
    dev_ = nullptr;
}

// ─── Texture::allocImage ────────────────────────────────────────────────────

void Texture::allocImage(RHIDevice& dev, uint32_t w, uint32_t h,
                          uint32_t mips, uint32_t layers,
                          VkFormat fmt, VkImageUsageFlags usage,
                          VkImageCreateFlags flags, VkImageViewType viewType)
{
    dev_       = &dev;
    width_     = w;
    height_    = h;
    mipLevels_ = mips;
    layers_    = layers;
    format_    = fmt;
    layout_    = VK_IMAGE_LAYOUT_UNDEFINED;

    // 创建 VkImage（复用 vk_helpers 的 createImage）
    vulkan_tutorial::createImage(
        dev.physicalDevice(), dev.device(),
        w, h, fmt,
        VK_IMAGE_TILING_OPTIMAL, usage,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        image_, memory_,
        mips, layers, flags);

    // 确定深度/颜色 Aspect
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    if (fmt == VK_FORMAT_D32_SFLOAT ||
        fmt == VK_FORMAT_D32_SFLOAT_S8_UINT ||
        fmt == VK_FORMAT_D24_UNORM_S8_UINT) {
        aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (fmt == VK_FORMAT_D32_SFLOAT_S8_UINT ||
            fmt == VK_FORMAT_D24_UNORM_S8_UINT)
            aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    // 创建 VkImageView
    VkImageViewCreateInfo viewCI{};
    viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image                           = image_;
    viewCI.viewType                        = viewType;
    viewCI.format                          = fmt;
    viewCI.subresourceRange.aspectMask     = aspect;
    viewCI.subresourceRange.baseMipLevel   = 0;
    viewCI.subresourceRange.levelCount     = mips;
    viewCI.subresourceRange.baseArrayLayer = 0;
    viewCI.subresourceRange.layerCount     = layers;
    VK_CHECK(vkCreateImageView(dev.device(), &viewCI, nullptr, &view_));
}

// ─── Texture::createSampler ─────────────────────────────────────────────────

void Texture::createSampler(RHIDevice& dev, const SamplerDesc& sd)
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(dev.physicalDevice(), &props);

    VkSamplerCreateInfo samplerCI{};
    samplerCI.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCI.magFilter        = sd.magFilter;
    samplerCI.minFilter        = sd.minFilter;
    samplerCI.addressModeU     = sd.addressMode;
    samplerCI.addressModeV     = sd.addressMode;
    samplerCI.addressModeW     = sd.addressMode;
    samplerCI.anisotropyEnable = (sd.enableAniso && dev.supportsFeature(DeviceFeature::SamplerAnisotropy))
                                     ? VK_TRUE : VK_FALSE;
    samplerCI.maxAnisotropy    = samplerCI.anisotropyEnable
                                     ? props.limits.maxSamplerAnisotropy : 1.0f;
    samplerCI.borderColor      = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerCI.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCI.minLod           = 0.0f;
    samplerCI.maxLod           = sd.enableMip ? sd.maxLod : 0.0f;
    samplerCI.mipLodBias       = 0.0f;
    VK_CHECK(vkCreateSampler(dev.device(), &samplerCI, nullptr, &sampler_));
}

// ─── Texture::transitionLayout ──────────────────────────────────────────────

void Texture::transitionLayout(VkCommandBuffer cmd,
                                VkImageLayout oldL, VkImageLayout newL,
                                uint32_t mips, uint32_t layers)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = oldL;
    barrier.newLayout                       = newL;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = image_;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = mips;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = layers;

    // Aspect 判断
    if (newL == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ||
        newL == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (format_ == VK_FORMAT_D32_SFLOAT_S8_UINT ||
            format_ == VK_FORMAT_D24_UNORM_S8_UINT)
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    } else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    VkPipelineStageFlags srcStage = 0, dstStage = 0;

    if (oldL == VK_IMAGE_LAYOUT_UNDEFINED &&
        newL == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldL == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newL == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldL == VK_IMAGE_LAYOUT_UNDEFINED &&
               newL == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                              | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else if (oldL == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newL == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldL == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
               newL == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    layout_ = newL;
}

// ─── Texture2D::generateMips ────────────────────────────────────────────────

void Texture2D::generateMips(VkCommandBuffer cmd)
{
    // 检查线性过滤支持
    VkFormatProperties fmtProps{};
    vkGetPhysicalDeviceFormatProperties(dev_->physicalDevice(), format_, &fmtProps);
    if (!(fmtProps.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
        transitionLayout(cmd,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            mipLevels_, 1);
        return;
    }

    VkImageMemoryBarrier barrier{};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image                           = image_;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    barrier.subresourceRange.levelCount     = 1;

    int32_t mipW = static_cast<int32_t>(width_);
    int32_t mipH = static_cast<int32_t>(height_);

    for (uint32_t i = 1; i < mipLevels_; ++i) {
        // 将 i-1 层从 TRANSFER_DST 转换为 TRANSFER_SRC
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        // Blit 到下一层
        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipW, mipH, 1};
        blit.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel       = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount     = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mipW > 1 ? mipW / 2 : 1, mipH > 1 ? mipH / 2 : 1, 1};
        blit.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel       = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount     = 1;

        vkCmdBlitImage(cmd,
            image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        // 将 i-1 层转为 SHADER_READ_ONLY
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        if (mipW > 1) mipW /= 2;
        if (mipH > 1) mipH /= 2;
    }

    // 最后一层也需要转换
    barrier.subresourceRange.baseMipLevel = mipLevels_ - 1;
    barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

// ─── Texture2D::loadFromFile ────────────────────────────────────────────────

void Texture2D::loadFromFile(RHIDevice& dev, const std::string& path,
                              bool srgb, bool genMips, const SamplerDesc& sd)
{
    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels)
        throw std::runtime_error("[Texture2D] 无法加载图像: " + path +
                                 "\n  " + stbi_failure_reason());

    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(w) * h * 4;
    const VkFormat fmt = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

    uint32_t mips = 1;
    if (genMips)
        mips = static_cast<uint32_t>(
            std::floor(std::log2(std::max(w, h)))) + 1;

    const VkImageUsageFlags usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;
    allocImage(dev, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
               mips, 1, fmt, usage);

    // 创建常驻映射的 staging buffer，并写入像素数据
    Buffer staging;
    {
        Buffer::CreateInfo sci{};
        sci.size          = imageSize;
        sci.usage         = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        sci.memProps      = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        sci.persistentMap = true;
        staging.create(dev, sci);
    }
    std::memcpy(staging.mapped(), pixels, static_cast<size_t>(imageSize));
    stbi_image_free(pixels);

    // 一次性命令：从 staging 复制到 image，再生成 mip
    VkCommandBuffer cmd = dev.beginOneShot();

    transitionLayout(cmd, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mips, 1);

    VkBufferImageCopy region{};
    region.bufferOffset                    = 0;
    region.bufferRowLength                 = 0;
    region.bufferImageHeight               = 0;
    region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel       = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount     = 1;
    region.imageOffset                     = {0, 0, 0};
    region.imageExtent                     = {static_cast<uint32_t>(w),
                                              static_cast<uint32_t>(h), 1};
    vkCmdCopyBufferToImage(cmd, staging.handle(), image_,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    if (genMips && mips > 1)
        generateMips(cmd);
    else
        transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1);

    dev.endOneShot(cmd);

    createSampler(dev, sd);
}

// ─── Texture2D::create ──────────────────────────────────────────────────────

void Texture2D::create(RHIDevice& dev, const CreateInfo& ci)
{
    uint32_t mips = 1;
    if (ci.genMips && ci.pixels)
        mips = static_cast<uint32_t>(
            std::floor(std::log2(std::max(ci.width, ci.height)))) + 1;

    const VkImageUsageFlags usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;
    allocImage(dev, ci.width, ci.height, mips, 1, ci.format, usage);

    if (ci.pixels) {
        const VkDeviceSize imageSize =
            static_cast<VkDeviceSize>(ci.width) * ci.height * 4;

        Buffer staging;
        Buffer::CreateInfo sci{};
        sci.size          = imageSize;
        sci.usage         = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        sci.memProps      = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        sci.persistentMap = true;
        staging.create(dev, sci);
        std::memcpy(staging.mapped(), ci.pixels, static_cast<size_t>(imageSize));

        VkCommandBuffer cmd = dev.beginOneShot();
        transitionLayout(cmd, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mips, 1);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel       = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount     = 1;
        region.imageExtent = {ci.width, ci.height, 1};
        vkCmdCopyBufferToImage(cmd, staging.handle(), image_,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        if (ci.genMips && mips > 1)
            generateMips(cmd);
        else
            transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1);
        dev.endOneShot(cmd);
    }

    createSampler(dev, ci.sampler);
}

// ─── TextureCube::create ─────────────────────────────────────────────────────

void TextureCube::create(RHIDevice& dev, const CreateInfo& ci)
{
    allocImage(dev, ci.size, ci.size, 1, 6, ci.format, ci.usage,
               VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
               VK_IMAGE_VIEW_TYPE_CUBE);

    // 创建每个面的独立 View
    for (uint32_t face = 0; face < 6; ++face) {
        VkImageViewCreateInfo viewCI{};
        viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCI.image                           = image_;
        viewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewCI.format                          = ci.format;
        viewCI.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCI.subresourceRange.baseMipLevel   = 0;
        viewCI.subresourceRange.levelCount     = 1;
        viewCI.subresourceRange.baseArrayLayer = face;
        viewCI.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(dev.device(), &viewCI, nullptr, &faceViews_[face]));
    }

    createSampler(dev, ci.sampler);
}

void TextureCube::destroyFaceViews()
{
    if (!dev_) return;
    for (auto& fv : faceViews_) {
        if (fv != VK_NULL_HANDLE) {
            vkDestroyImageView(dev_->device(), fv, nullptr);
            fv = VK_NULL_HANDLE;
        }
    }
}

// ─── RenderTarget::create ───────────────────────────────────────────────────

void RenderTarget::create(RHIDevice& dev, const CreateInfo& ci)
{
    type_ = ci.type;

    VkFormat fmt = ci.format;
    VkImageUsageFlags usage = 0;

    if (ci.type == Type::Color) {
        if (fmt == VK_FORMAT_UNDEFINED)
            fmt = VK_FORMAT_R8G8B8A8_UNORM;
        usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (ci.needSampling)
            usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    } else {
        if (fmt == VK_FORMAT_UNDEFINED)
            fmt = dev.depthFormat();
        usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (ci.needSampling)
            usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }

    allocImage(dev, ci.width, ci.height, 1, 1, fmt, usage);

    // 初始化布局
    VkCommandBuffer cmd = dev.beginOneShot();
    VkImageLayout targetLayout = (ci.type == Type::Color)
        ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    transitionLayout(cmd, VK_IMAGE_LAYOUT_UNDEFINED, targetLayout, 1, 1);
    dev.endOneShot(cmd);

    if (ci.needSampling)
        createSampler(dev, ci.sampler);
}

// ─── TextureCache ───────────────────────────────────────────────────────────

void TextureCache::destroy()
{
    cache_.clear();
    dev_ = nullptr;
}

Texture2D& TextureCache::load(const std::string& path, bool srgb, bool genMips)
{
    auto it = cache_.find(path);
    if (it != cache_.end()) {
        ++stats_.cacheHits;
        return *it->second;
    }

    ++stats_.cacheMisses;
    ++stats_.totalLoaded;
    auto tex = std::make_unique<Texture2D>();
    tex->loadFromFile(*dev_, path, srgb, genMips);
    auto& ref = *tex;
    cache_.emplace(path, std::move(tex));
    return ref;
}

void TextureCache::unload(const std::string& path)
{
    cache_.erase(path);
}

void TextureCache::unloadAll()
{
    cache_.clear();
}

} // namespace engine
