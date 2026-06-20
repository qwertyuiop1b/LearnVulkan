#pragma once
/**
 * @file rhi_texture.hpp
 * @brief 第63章：纹理系统
 *
 * 类层次：
 *   Texture        — 基础 VkImage + View + Sampler RAII 包装
 *   Texture2D      — 从文件或内存创建，自动生成 Mip
 *   TextureCube    — Cubemap（6 层）
 *   RenderTarget   — 可作为 Attachment 和 Sampler 的纹理
 *   TextureCache   — 路径 → Texture2D 缓存，防止重复加载
 */

#include "rhi_device.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace engine {

// ─── 采样器描述符 ───────────────────────────────────────────────────────────

struct SamplerDesc {
    VkFilter magFilter = VK_FILTER_LINEAR;
    VkFilter minFilter = VK_FILTER_LINEAR;
    VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    bool enableAniso = true;
    bool enableMip = true;
    float maxLod = 16.0f;
};

// ─── 基础纹理 ───────────────────────────────────────────────────────────────

/**
 * @brief GPU 纹理 RAII 包装
 *
 * 持有：VkImage + VkDeviceMemory + VkImageView + VkSampler
 */
class Texture {
  public:
    Texture() = default;
    virtual ~Texture() {
        destroy();
    }
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&) noexcept;
    Texture& operator=(Texture&&) noexcept;

    void destroy();

    [[nodiscard]] VkImage image() const {
        return image_;
    }
    [[nodiscard]] VkImageView view() const {
        return view_;
    }
    [[nodiscard]] VkSampler sampler() const {
        return sampler_;
    }
    [[nodiscard]] VkFormat format() const {
        return format_;
    }
    [[nodiscard]] uint32_t width() const {
        return width_;
    }
    [[nodiscard]] uint32_t height() const {
        return height_;
    }
    [[nodiscard]] uint32_t mipLevels() const {
        return mipLevels_;
    }
    [[nodiscard]] bool isValid() const {
        return image_ != VK_NULL_HANDLE;
    }

    [[nodiscard]] VkDescriptorImageInfo descriptorInfo() const {
        return {sampler_, view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }

  protected:
    void allocImage(RHIDevice& dev,
                    uint32_t w,
                    uint32_t h,
                    uint32_t mips,
                    uint32_t layers,
                    VkFormat fmt,
                    VkImageUsageFlags usage,
                    VkImageCreateFlags flags = 0,
                    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D);
    void createSampler(RHIDevice& dev, const SamplerDesc& sd);
    void transitionLayout(
        VkCommandBuffer cmd, VkImageLayout oldL, VkImageLayout newL, uint32_t mips = 1, uint32_t layers = 1);

    RHIDevice* dev_ = nullptr;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t mipLevels_ = 1;
    uint32_t layers_ = 1;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
};

// ─── Texture2D ──────────────────────────────────────────────────────────────

/**
 * @brief 2D 纹理（最常用的类型）
 *
 * 支持：
 *   - 从文件路径加载（stb_image）
 *   - 从原始像素数据创建
 *   - 自动生成 Mip（使用 blit 命令）
 *   - sRGB / Linear 自动选择
 */
class Texture2D : public Texture {
  public:
    struct CreateInfo {
        uint32_t width = 0;
        uint32_t height = 0;
        VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
        const void* pixels = nullptr;
        bool genMips = true;
        SamplerDesc sampler{};
    };

    /// 从文件加载（stb_image 读取 → 上传）
    void loadFromFile(
        RHIDevice& dev, const std::string& path, bool srgb = true, bool genMips = true, const SamplerDesc& sd = {});

    /// 从内存创建
    void create(RHIDevice& dev, const CreateInfo& ci);

  private:
    void generateMips(VkCommandBuffer cmd);
};

// ─── TextureCube ────────────────────────────────────────────────────────────

/**
 * @brief Cubemap（6 个面）
 *
 * 用于：天空盒、反射探针。
 * 每面尺寸必须相同，格式相同。
 */
class TextureCube : public Texture {
  public:
    struct CreateInfo {
        uint32_t size = 256; ///< 每面的宽/高（必须相等）
        VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
        VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        SamplerDesc sampler{};
    };

    void create(RHIDevice& dev, const CreateInfo& ci);

    /// 获取某一面的 ImageView（用于渲染捕获）
    [[nodiscard]] VkImageView faceView(uint32_t face) const {
        return faceViews_[face];
    }

    void destroyFaceViews();

  private:
    VkImageView faceViews_[6] = {};
};

// ─── RenderTarget ───────────────────────────────────────────────────────────

/**
 * @brief 离屏渲染目标
 *
 * 同时支持：作为 Framebuffer 的 Color/Depth Attachment
 *            和作为后续 Pass 的 Sampled Texture
 */
class RenderTarget : public Texture {
  public:
    enum class Type { Color, Depth, DepthStencil };

    struct CreateInfo {
        uint32_t width = 0;
        uint32_t height = 0;
        Type type = Type::Color;
        VkFormat format = VK_FORMAT_UNDEFINED; ///< UNDEFINED = 自动选择
        SamplerDesc sampler{};
        bool needSampling = true; ///< 是否需要在着色器中采样
    };

    void create(RHIDevice& dev, const CreateInfo& ci);

    [[nodiscard]] bool isColor() const {
        return type_ == Type::Color;
    }
    [[nodiscard]] bool isDepth() const {
        return type_ == Type::Depth || type_ == Type::DepthStencil;
    }
    [[nodiscard]] VkDescriptorImageInfo descriptorInfo() const {
        VkImageLayout l =
            isDepth() ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return {sampler_, view_, l};
    }

  private:
    Type type_ = Type::Color;
};

// ─── TextureCache ───────────────────────────────────────────────────────────

/**
 * @brief 纹理缓存 —— 防止同一文件被加载两次
 *
 * 用法：
 * @code
 *   TextureCache cache;
 *   cache.init(dev);
 *   auto& tex = cache.load("assets/textures/brick.png");
 *   // 再次加载同路径：直接返回缓存的对象，不重新分配 GPU 内存
 *   auto& tex2 = cache.load("assets/textures/brick.png");
 *   assert(&tex == &tex2);   // 同一对象
 * @endcode
 */
class TextureCache {
  public:
    void init(RHIDevice& dev) {
        dev_ = &dev;
    }
    void destroy();

    [[nodiscard]] Texture2D& load(const std::string& path, bool srgb = true, bool genMips = true);
    void unload(const std::string& path);
    void unloadAll();

    [[nodiscard]] size_t count() const {
        return cache_.size();
    }
    [[nodiscard]] bool isCached(const std::string& path) const {
        return cache_.find(path) != cache_.end();
    }

    struct Stats {
        size_t totalLoaded = 0;
        size_t cacheHits = 0;
        size_t cacheMisses = 0;
    };
    [[nodiscard]] const Stats& stats() const {
        return stats_;
    }

  private:
    RHIDevice* dev_ = nullptr;
    std::unordered_map<std::string, std::unique_ptr<Texture2D>> cache_;
    Stats stats_{};
};

} // namespace engine
