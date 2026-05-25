#pragma once
/**
 * @file rhi_device.hpp
 * @brief 第61章：设备抽象层（RHI Device）
 *
 * 将裸 VkInstance / VkPhysicalDevice / VkDevice 封装成一个有意义的对象。
 * 好处：
 *   - 统一管理设备生命周期（RAII）
 *   - 能力查询接口比直接调用 Vulkan API 更清晰
 *   - 后续 Buffer / Texture / Pipeline 都接受 RHIDevice& 而不是散落的句柄
 */

#include <vulkan/vulkan.h>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace engine {

/// 设备特性枚举（游戏引擎常用的能力）
enum class DeviceFeature : uint32_t {
    GeometryShader,
    TessellationShader,
    WideLines,
    FillModeNonSolid,
    SamplerAnisotropy,
    MultiDrawIndirect,
    DrawIndirectFirstInstance,
    ShaderClipDistance,
    TimestampQueries,
    MeshShader,
    RayTracing,
};

/// 创建参数
struct DeviceCreateInfo {
    void*        windowHandle   = nullptr;   ///< GLFWwindow*，用于创建 Surface
    bool         enableValidation = true;
    bool         preferDiscreteGpu = true;   ///< 优先独显
    std::vector<DeviceFeature> requiredFeatures;
    std::vector<DeviceFeature> optionalFeatures;
    uint32_t     frameCount = 2;             ///< 并发帧数（影响 command pool 数量）
};

/// 队列族描述
struct QueueInfo {
    VkQueue  handle      = VK_NULL_HANDLE;
    uint32_t familyIndex = UINT32_MAX;
    bool     supportsTimestamps = false;
    float    timestampPeriodNs  = 0.0f;
};

/**
 * @brief 设备抽象 —— Vulkan 设备的高层封装
 *
 * 使用示例：
 * @code
 *   DeviceCreateInfo ci{};
 *   ci.windowHandle = window;
 *   ci.requiredFeatures = { DeviceFeature::SamplerAnisotropy };
 *   RHIDevice dev;
 *   dev.init(ci);
 *   // ... 使用 dev.device(), dev.graphicsQueue() 等
 *   dev.destroy();
 * @endcode
 */
class RHIDevice {
public:
    RHIDevice() = default;
    ~RHIDevice() { destroy(); }

    RHIDevice(const RHIDevice&) = delete;
    RHIDevice& operator=(const RHIDevice&) = delete;
    RHIDevice(RHIDevice&&) noexcept;
    RHIDevice& operator=(RHIDevice&&) noexcept;

    /// 初始化（等效于 ch01-ch03 的全部内容）
    void init(const DeviceCreateInfo& ci);
    void destroy();

    // ─── 访问底层句柄 ─────────────────────────────────────────────────────
    [[nodiscard]] VkInstance         instance()        const { return instance_; }
    [[nodiscard]] VkPhysicalDevice   physicalDevice()  const { return physDev_; }
    [[nodiscard]] VkDevice           device()          const { return device_; }
    [[nodiscard]] VkSurfaceKHR       surface()         const { return surface_; }
    [[nodiscard]] const QueueInfo&   graphicsQueue()   const { return graphics_; }
    [[nodiscard]] const QueueInfo&   computeQueue()    const { return compute_; }
    [[nodiscard]] const QueueInfo&   transferQueue()   const { return transfer_; }

    // ─── 能力查询 ─────────────────────────────────────────────────────────
    [[nodiscard]] bool supportsFeature(DeviceFeature f) const;
    [[nodiscard]] std::string deviceName() const { return deviceName_; }
    [[nodiscard]] bool isDiscreteGpu()     const { return isDiscrete_; }
    [[nodiscard]] VkDeviceSize totalVideoMemoryBytes() const { return videoMemBytes_; }

    // ─── 便利方法 ─────────────────────────────────────────────────────────
    [[nodiscard]] uint32_t findMemoryType(uint32_t typeBits,
                                          VkMemoryPropertyFlags props) const;
    [[nodiscard]] VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,
                                               VkImageTiling tiling,
                                               VkFormatFeatureFlags features) const;
    [[nodiscard]] VkFormat depthFormat() const { return depthFmt_; }
    void waitIdle() const { vkDeviceWaitIdle(device_); }

    // ─── 单次命令辅助 ─────────────────────────────────────────────────────
    [[nodiscard]] VkCommandBuffer beginOneShot();
    void                          endOneShot(VkCommandBuffer cmd);
    [[nodiscard]] VkCommandPool   commandPool() const { return cmdPool_; }

    [[nodiscard]] bool isValid() const { return device_ != VK_NULL_HANDLE; }

private:
    void pickPhysicalDevice(bool preferDiscrete);
    void createLogicalDevice(const std::vector<DeviceFeature>& req,
                             const std::vector<DeviceFeature>& opt);
    void detectCapabilities();
    void cacheDepthFormat();

    VkInstance       instance_  = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_   = VK_NULL_HANDLE;
    VkPhysicalDevice physDev_   = VK_NULL_HANDLE;
    VkDevice         device_    = VK_NULL_HANDLE;
    VkCommandPool    cmdPool_   = VK_NULL_HANDLE;

    QueueInfo graphics_{};
    QueueInfo compute_{};
    QueueInfo transfer_{};

    VkFormat     depthFmt_    = VK_FORMAT_UNDEFINED;
    std::string  deviceName_;
    bool         isDiscrete_  = false;
    VkDeviceSize videoMemBytes_ = 0;

    uint32_t enabledFeatureMask_ = 0;  // bitfield of DeviceFeature

    VkDebugUtilsMessengerEXT debugMsg_ = VK_NULL_HANDLE;
};

} // namespace engine
