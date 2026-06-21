/**
 * @file ch20_raytracing.cpp
 * @brief 第20章：硬件光线追踪（VK_KHR_ray_tracing_pipeline）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【光线追踪 vs 光栅化】
 *
 *  光栅化（当前所学）：
 *    几何体 → 投影到屏幕 → 光栅化为像素 → 着色
 *    复杂效果（反射、折射、全局光照）需要技巧模拟
 *
 *  光线追踪（Ray Tracing）：
 *    从相机发射光线 → 与场景求交 → 物理正确计算光照
 *    天然支持：反射、折射、阴影、环境光遮蔽（AO）
 *
 * 【硬件加速光线追踪（NVIDIA Turing+ / AMD RDNA2+）】
 *
 *  RT Core（专用硬件单元）加速：
 *    BVH（Bounding Volume Hierarchy）遍历
 *    光线-三角形/AABB 求交测试
 *
 * 【Vulkan Ray Tracing Pipeline 核心概念】
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  加速结构（Acceleration Structure）                      │
 *  │                                                         │
 *  │  BLAS（Bottom Level AS）                                │
 *  │    → 每个几何体的三角形网格                              │
 *  │    → vkCreateAccelerationStructureKHR                   │
 *  │    → vkCmdBuildAccelerationStructuresKHR                │
 *  │                                                         │
 *  │  TLAS（Top Level AS）                                   │
 *  │    → 场景中所有 BLAS 实例（含变换矩阵）                  │
 *  │    → 相当于光线追踪的"场景图"                           │
 *  └─────────────────────────────────────────────────────────┘
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  新增着色器类型                                          │
 *  │                                                         │
 *  │  Ray Generation Shader (.rgen)                         │
 *  │    → 为每个像素发射光线                                  │
 *  │    → traceRayEXT(tlas, ...) 触发求交                    │
 *  │                                                         │
 *  │  Miss Shader (.rmiss)                                   │
 *  │    → 光线未命中任何几何体时调用                          │
 *  │    → 通常返回天空盒颜色                                  │
 *  │                                                         │
 *  │  Closest Hit Shader (.rchit)                           │
 *  │    → 光线命中最近的几何体时调用                          │
 *  │    → 计算光照、发射次级光线（递归）                      │
 *  │                                                         │
 *  │  Any Hit Shader (.rahit) [可选]                        │
 *  │    → 光线命中任意几何体时调用（用于 Alpha Test）         │
 *  │                                                         │
 *  │  Intersection Shader (.rint) [可选]                    │
 *  │    → 自定义求交（用于球体、曲线等非三角形几何）           │
 *  └─────────────────────────────────────────────────────────┘
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  Shader Binding Table（SBT）着色器绑定表                │
 *  │                                                         │
 *  │  描述光线追踪时如何选择着色器的规则表：                   │
 *  │    [rgen handle]                                        │
 *  │    [miss handle 0] [miss handle 1]                      │
 *  │    [hit handle 0 for geometry 0]                        │
 *  │    [hit handle 1 for geometry 1]                        │
 *  │    ...                                                  │
 *  └─────────────────────────────────────────────────────────┘
 *
 * 【vkCmdTraceRaysKHR 替代 vkCmdDraw】
 *
 *  光线追踪不使用图形管线，而是：
 *    vkCmdBindPipeline(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline)
 *    vkCmdTraceRaysKHR(cmd, &rgenSBT, &missSBT, &hitSBT, &callableSBT,
 *                       width, height, depth)
 *  → 每个像素发射一条光线（width×height 条）
 *
 * 【所需扩展】
 *
 *  VK_KHR_ray_tracing_pipeline
 *  VK_KHR_acceleration_structure
 *  VK_KHR_buffer_device_address（获取缓冲 GPU 地址）
 *  VK_KHR_deferred_host_operations（异步 AS 构建）
 *  VK_KHR_spirv_1_4 + VK_KHR_shader_float_controls
 *
 * 【macOS / MoltenVK 支持情况】
 *
 *  ⚠️  MoltenVK 目前不支持硬件光线追踪（Metal 3 支持有限）
 *  ⚠️  本章代码在 NVIDIA/AMD GPU + Windows/Linux 上运行
 *  ✅  代码结构正确，可用于理解光线追踪 API
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <cstring>
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#include <vulkan_tutorial/utils.hpp>

// ─── 光线追踪所需的额外扩展
// ───────────────────────────────────────────────────

static const std::vector<const char*> RT_DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    "VK_KHR_ray_tracing_pipeline",     // 光线追踪管线
    "VK_KHR_acceleration_structure",   // 加速结构
    "VK_KHR_buffer_device_address",    // GPU 缓冲地址
    "VK_KHR_deferred_host_operations", // 异步 AS 构建
    "VK_KHR_spirv_1_4",                // SPIR-V 1.4
    "VK_KHR_shader_float_controls",    // 浮点控制
};

// ─── 光线追踪管线属性
// ──────────────────────────────────────────────────────────

struct RayTracingProperties {
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR pipeline;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accel;
};

// ─── GLSL 光线追踪着色器（内嵌简单版本） ─────────────────────────────────────
//
// 完整的光线追踪着色器需要 SPIR-V 1.4 支持：
//
// ray_gen.rgen：
//   #version 460
//   #extension GL_EXT_ray_tracing : require
//
//   layout(binding = 0, set = 0) uniform accelerationStructureEXT topAS;
//   layout(binding = 1, set = 0, rgba8) uniform image2D outputImage;
//   layout(binding = 2, set = 0) uniform Camera { mat4 viewInverse; mat4
//   projInverse; } cam;
//
//   layout(location = 0) rayPayloadEXT vec3 hitValue;
//
//   void main() {
//       vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + 0.5;
//       vec2 uv = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
//       vec2 d = uv * 2.0 - 1.0;
//
//       vec4 origin    = cam.viewInverse * vec4(0, 0, 0, 1);
//       vec4 target    = cam.projInverse * vec4(d.x, d.y, 1, 1);
//       vec4 direction = cam.viewInverse * vec4(normalize(target.xyz), 0);
//
//       traceRayEXT(topAS, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0,
//                   origin.xyz, 0.001, direction.xyz, 10000.0, 0);
//
//       imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue, 1.0));
//   }
//
// closest_hit.rchit：
//   #extension GL_EXT_ray_tracing : require
//   layout(location = 0) rayPayloadInEXT vec3 hitValue;
//   hitAttributeEXT vec2 attribs;   // 重心坐标
//
//   void main() {
//       vec3 barycentrics = vec3(1 - attribs.x - attribs.y, attribs.x,
//       attribs.y); hitValue = barycentrics;   //
//       用重心坐标作为颜色（彩色三角形）
//   }
//
// miss.rmiss：
//   #extension GL_EXT_ray_tracing : require
//   layout(location = 0) rayPayloadInEXT vec3 hitValue;
//   void main() { hitValue = vec3(0.1, 0.2, 0.4); }  // 天空蓝色

/**
 * @brief 光线追踪应用（演示 API 结构，需要 NVIDIA/AMD 支持的硬件）
 */
class Ch20App {
  public:
    void run() {
        initWindow();

        // 检查光线追踪支持
        if (!initVulkan()) {
            std::cout << "\n⚠️  此设备不支持光线追踪，请查看代码了解 API "
                         "结构。\n";
            printRayTracingGuide();
            glfwDestroyWindow(window_);
            glfwTerminate();
            return;
        }

        std::cout << "\n🔭 光线追踪管线已初始化！开始渲染...\n";
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
    VkCommandPool commandPool_ = VK_NULL_HANDLE;

    // ── 光线追踪管线 ───────────────────────────────────────────────────────
    VkDescriptorSetLayout rtDescSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool      rtDescPool_      = VK_NULL_HANDLE;
    VkDescriptorSet       rtDescSet_       = VK_NULL_HANDLE;
    VkPipelineLayout rtPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline rtPipeline_ = VK_NULL_HANDLE;

    // ── SBT 区域（由 createShaderBindingTable 填写） ────────────────────
    VkStridedDeviceAddressRegionKHR sbtRgen_{}, sbtMiss_{}, sbtHit_{}, sbtCall_{};

    // ── 加速结构 ───────────────────────────────────────────────────────────
    VkAccelerationStructureKHR blas_ = VK_NULL_HANDLE; ///< 三角形网格
    VkAccelerationStructureKHR tlas_ = VK_NULL_HANDLE; ///< 场景实例
    VkBuffer blasBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory blasMemory_ = VK_NULL_HANDLE;
    VkBuffer tlasBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory tlasMemory_ = VK_NULL_HANDLE;

    // ── Shader Binding Table ──────────────────────────────────────────────
    VkBuffer sbtBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory sbtMemory_ = VK_NULL_HANDLE;

    // ── 材质 SSBO ─────────────────────────────────────────────────────────
    VkBuffer materialBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory materialMemory_ = VK_NULL_HANDLE;

    // ── 输出图像（光线追踪结果写入这里，再 blit 到交换链） ────────────────
    VkImage rtOutputImage_ = VK_NULL_HANDLE;
    VkDeviceMemory rtOutputMemory_ = VK_NULL_HANDLE;
    VkImageView rtOutputView_ = VK_NULL_HANDLE;

    // ── 累积图像（rgba32f，跨帧路径追踪累积） ────────────────────────────
    VkImage        accumImage_        = VK_NULL_HANDLE;
    VkDeviceMemory accumMemory_       = VK_NULL_HANDLE;
    VkImageView    accumView_         = VK_NULL_HANDLE;
    uint32_t       accumFrameCount_   = 0;

    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    QueueFamilyIndices queueIndices_;

    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence> inFlightFences_;
    std::vector<VkCommandBuffer> commandBuffers_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;

    // 函数指针（扩展函数需要手动加载）
    PFN_vkCreateAccelerationStructureKHR fpCreateAccelStructure_ = nullptr;
    PFN_vkDestroyAccelerationStructureKHR fpDestroyAccelStructure_ = nullptr;
    PFN_vkBuildAccelerationStructuresKHR fpBuildAccelStructures_ = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR fpCmdBuildAccelStructures_ = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR fpGetAccelBuildSizes_ = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR fpGetAccelDeviceAddress_ = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR fpCreateRTPipelines_ = nullptr;
    PFN_vkCmdTraceRaysKHR fpCmdTraceRays_ = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR fpGetRTShaderGroupHandles_ = nullptr;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(800, 600, "Ch20 - 光线追踪（需要 RTX/RDNA2+ GPU）", nullptr, nullptr);
    }

    bool initVulkan() {
        try {
            createInstance();
            createSurface();
            if (!pickPhysicalDeviceRT())
                return false; // 检查 RT 支持
            createLogicalDeviceRT();
            loadRTFunctionPointers();
            createSwapchain();
            createImageViews();
            createCommandPool();
            createRTOutputImage();
            createAccumImage();
            buildAccelerationStructures();
            createRTPipeline();
            createShaderBindingTable();
            createCommandBuffers();
            createSyncObjects();
        } catch (const std::exception& e) {
            std::cerr << "⚠️  光线追踪初始化失败：" << e.what() << "\n";
            return false;
        }
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 检查光线追踪支持
    // ═══════════════════════════════════════════════════════════════════════

    bool pickPhysicalDeviceRT() {
        uint32_t c = 0;
        vkEnumeratePhysicalDevices(instance_, &c, nullptr);
        std::vector<VkPhysicalDevice> devs(c);
        vkEnumeratePhysicalDevices(instance_, &c, devs.data());

        for (auto& d : devs) {
            if (!findQueueFamilies(d, surface_).isComplete())
                continue;

            // 检查光线追踪扩展支持
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, exts.data());

            bool hasRTExt = false;
            for (auto& ext : exts)
                if (std::string(ext.extensionName) == "VK_KHR_ray_tracing_pipeline") {
                    hasRTExt = true;
                    break;
                }

            if (!hasRTExt)
                continue;

            // 检查光线追踪特性
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
            rtFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &rtFeatures;
            vkGetPhysicalDeviceFeatures2(d, &features2);

            if (!rtFeatures.rayTracingPipeline)
                continue;

            physicalDevice_ = d;
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);
            std::cout << "✅ 发现支持光线追踪的 GPU：" << p.deviceName << "\n";

            // 打印光线追踪管线属性
            VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
            rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &rtProps;
            vkGetPhysicalDeviceProperties2(d, &props2);

            std::cout << "🔭 最大递归深度：" << rtProps.maxRayRecursionDepth << "\n";
            std::cout << "📐 着色器组句柄大小：" << rtProps.shaderGroupHandleSize << " 字节\n";
            return true;
        }

        std::cout << "⚠️  未找到支持 VK_KHR_ray_tracing_pipeline 的 GPU\n";
        std::cout << "   请使用 NVIDIA RTX 或 AMD RDNA2+ 显卡\n";
        return false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 创建支持光线追踪的逻辑设备
    // ═══════════════════════════════════════════════════════════════════════

    void createLogicalDeviceRT() {
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

        // ── 启用光线追踪相关特性 ─────────────────────────────────────────
        VkPhysicalDeviceBufferDeviceAddressFeatures bufAddrFeatures{};
        bufAddrFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        bufAddrFeatures.bufferDeviceAddress = VK_TRUE;

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
        rtFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        rtFeatures.rayTracingPipeline = VK_TRUE;
        rtFeatures.pNext = &bufAddrFeatures;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
        asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        asFeatures.accelerationStructure = VK_TRUE;
        asFeatures.pNext = &rtFeatures;

        // 动态构建扩展列表，VK_KHR_portability_subset 仅在设备支持时添加（macOS）
        std::vector<const char*> enabledExts(RT_DEVICE_EXTENSIONS.begin(), RT_DEVICE_EXTENSIONS.end());
        {
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> availExts(extCount);
            vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, availExts.data());
            for (auto& e : availExts)
                if (std::string(e.extensionName) == "VK_KHR_portability_subset")
                    enabledExts.push_back("VK_KHR_portability_subset");
        }

        VkPhysicalDeviceFeatures feat{};
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.pNext = &asFeatures; // pNext 链：AS → RT Pipeline → Buffer Device Address
        ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos = qcis.data();
        ci.pEnabledFeatures = &feat;
        ci.enabledExtensionCount = static_cast<uint32_t>(enabledExts.size());
        ci.ppEnabledExtensionNames = enabledExts.data();
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_));
        vkGetDeviceQueue(device_, queueIndices_.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueIndices_.presentFamily.value(), 0, &presentQueue_);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 加载光线追踪扩展函数指针（必须手动加载，不在 Vulkan 核心中）
    // ═══════════════════════════════════════════════════════════════════════

    void loadRTFunctionPointers() {
        auto load = [this](const char* name, void** pfn) {
            *pfn = reinterpret_cast<void*>(vkGetDeviceProcAddr(device_, name));
            if (!*pfn)
                throw std::runtime_error(std::string("无法加载 ") + name);
        };

        load("vkCreateAccelerationStructureKHR", reinterpret_cast<void**>(&fpCreateAccelStructure_));
        load("vkDestroyAccelerationStructureKHR", reinterpret_cast<void**>(&fpDestroyAccelStructure_));
        load("vkBuildAccelerationStructuresKHR", reinterpret_cast<void**>(&fpBuildAccelStructures_));
        load("vkCmdBuildAccelerationStructuresKHR", reinterpret_cast<void**>(&fpCmdBuildAccelStructures_));
        load("vkGetAccelerationStructureBuildSizesKHR", reinterpret_cast<void**>(&fpGetAccelBuildSizes_));
        load("vkGetAccelerationStructureDeviceAddressKHR", reinterpret_cast<void**>(&fpGetAccelDeviceAddress_));
        load("vkCreateRayTracingPipelinesKHR", reinterpret_cast<void**>(&fpCreateRTPipelines_));
        load("vkCmdTraceRaysKHR", reinterpret_cast<void**>(&fpCmdTraceRays_));
        load("vkGetRayTracingShaderGroupHandlesKHR", reinterpret_cast<void**>(&fpGetRTShaderGroupHandles_));

        std::cout << "✅ 光线追踪扩展函数指针加载完成\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 构建加速结构（球体 BLAS + 多实例 TLAS + 材质 SSBO）
    // ═══════════════════════════════════════════════════════════════════════

    // GPU 材质（std430 对齐，32字节）
    struct MaterialGPU {
        float albedo[4]; // xyz=albedo, w=fuzz(Metal)/ior(Dielectric)
        int32_t type;    // 0=Lambertian, 1=Metal, 2=Dielectric
        int32_t pad[3];
    };

    // 生成单位球体网格（lat×lon 细分）
    void genUnitSphere(int nlat, int nlon,
                       std::vector<float>& verts, std::vector<uint32_t>& idx) {
        for (int i = 0; i <= nlat; ++i) {
            float phi = 3.14159265f * i / nlat;
            for (int j = 0; j <= nlon; ++j) {
                float theta = 2.f * 3.14159265f * j / nlon;
                verts.push_back(std::sin(phi) * std::cos(theta));
                verts.push_back(std::cos(phi));
                verts.push_back(std::sin(phi) * std::sin(theta));
            }
        }
        for (int i = 0; i < nlat; ++i)
            for (int j = 0; j < nlon; ++j) {
                uint32_t a = i * (nlon + 1) + j;
                uint32_t b = a + 1;
                uint32_t c = a + (nlon + 1);
                uint32_t d = c + 1;
                idx.insert(idx.end(), {a, c, b, b, c, d});
            }
    }

    // 构造 3×4 行主序变换矩阵（均匀缩放 + 平移）
    static VkTransformMatrixKHR makeTransform(float s, float tx, float ty, float tz) {
        VkTransformMatrixKHR m{};
        m.matrix[0][0] = s;  m.matrix[0][3] = tx;
        m.matrix[1][1] = s;  m.matrix[1][3] = ty;
        m.matrix[2][2] = s;  m.matrix[2][3] = tz;
        return m;
    }

    // 简单整数哈希，用于确定性随机场景生成
    static float ihash(int n) {
        n = (n ^ 61) ^ (n >> 16);
        n += (n << 3);
        n ^= (n >> 4);
        n *= 0x27d4eb2d;
        n ^= (n >> 15);
        return float(n & 0x7fffffff) / float(0x7fffffff);
    }

    void buildAccelerationStructures() {
        // ── 生成单位球体网格（16×16 细分） ────────────────────────────────
        std::vector<float>    sphereVerts;
        std::vector<uint32_t> sphereIdx;
        genUnitSphere(16, 16, sphereVerts, sphereIdx);
        const uint32_t numSphereVerts = static_cast<uint32_t>(sphereVerts.size() / 3);
        const uint32_t numSphereTris  = static_cast<uint32_t>(sphereIdx.size() / 3);

        // ── 创建顶点/索引缓冲（BLAS 构建时读取） ─────────────────────────
        VkBuffer vertBuf, idxBuf;
        VkDeviceMemory vertMem, idxMem;
        VkDeviceSize vertBytes = sphereVerts.size() * sizeof(float);
        VkDeviceSize idxBytes  = sphereIdx.size()  * sizeof(uint32_t);
        createBufferForAS(vertBytes,
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                          sphereVerts.data(), vertBytes, vertBuf, vertMem);
        createBufferForAS(idxBytes,
                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                          sphereIdx.data(), idxBytes, idxBuf, idxMem);

        VkDeviceAddress vertAddr = getBufferDeviceAddress(vertBuf);
        VkDeviceAddress idxAddr  = getBufferDeviceAddress(idxBuf);

        // ── BLAS 几何描述（单位球体） ─────────────────────────────────────
        VkAccelerationStructureGeometryKHR geom{};
        geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geom.geometry.triangles.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geom.geometry.triangles.vertexData.deviceAddress = vertAddr;
        geom.geometry.triangles.vertexStride = sizeof(float) * 3;
        geom.geometry.triangles.maxVertex    = numSphereVerts - 1;
        geom.geometry.triangles.indexType    = VK_INDEX_TYPE_UINT32;
        geom.geometry.triangles.indexData.deviceAddress = idxAddr;

        // ── 查询 BLAS 构建所需的内存大小 ──────────────────────────────────
        VkAccelerationStructureBuildGeometryInfoKHR blasInfo{};
        blasInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        blasInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        blasInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        blasInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        blasInfo.geometryCount = 1;
        blasInfo.pGeometries   = &geom;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        fpGetAccelBuildSizes_(
            device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasInfo, &numSphereTris, &sizeInfo);

        // ── 创建 BLAS 缓冲和对象 ──────────────────────────────────────────
        createBuffer(sizeInfo.accelerationStructureSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     blasBuffer_,
                     blasMemory_);

        VkAccelerationStructureCreateInfoKHR blasCI{};
        blasCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        blasCI.buffer = blasBuffer_;
        blasCI.size = sizeInfo.accelerationStructureSize;
        blasCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        VK_CHECK(fpCreateAccelStructure_(device_, &blasCI, nullptr, &blas_));

        // Scratch 缓冲（AS 构建时需要的临时工作区）
        VkBuffer scratchBuf;
        VkDeviceMemory scratchMem;
        createBuffer(sizeInfo.buildScratchSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     scratchBuf,
                     scratchMem);

        // ── 提交 BLAS 构建命令 ────────────────────────────────────────────
        blasInfo.dstAccelerationStructure = blas_;
        blasInfo.scratchData.deviceAddress = getBufferDeviceAddress(scratchBuf);

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{numSphereTris, 0, 0, 0};
        const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

        VkCommandBuffer cmd = beginSingleTimeCommands();
        fpCmdBuildAccelStructures_(cmd, 1, &blasInfo, &pRangeInfo);
        endSingleTimeCommands(cmd);

        // 获取 BLAS 的设备地址（用于 TLAS 实例引用）
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = blas_;
        VkDeviceAddress blasAddr = fpGetAccelDeviceAddress_(device_, &addrInfo);

        // ── 构建 RTIOW 场景实例 + 材质数据 ─────────────────────────────────
        std::vector<VkAccelerationStructureInstanceKHR> instances;
        std::vector<MaterialGPU> materials;

        auto addSphere = [&](float s, float tx, float ty, float tz,
                             float r, float g, float b, float fw, int type) {
            MaterialGPU m{};
            m.albedo[0] = r; m.albedo[1] = g; m.albedo[2] = b; m.albedo[3] = fw;
            m.type = type;
            uint32_t matIdx = static_cast<uint32_t>(materials.size());
            materials.push_back(m);

            VkAccelerationStructureInstanceKHR inst{};
            inst.transform         = makeTransform(s, tx, ty, tz);
            inst.instanceCustomIndex    = matIdx;
            inst.mask                   = 0xFF;
            inst.instanceShaderBindingTableRecordOffset = 0;
            inst.flags                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            inst.accelerationStructureReference = blasAddr;
            instances.push_back(inst);
        };

        // 地面（大球）
        addSphere(1000.f, 0.f, -1000.f, 0.f,  0.5f, 0.5f, 0.5f, 0.f, 0); // Lambert gray
        // 三个特写球
        addSphere(1.f,  0.f, 1.f, 0.f,  0.f, 0.f, 0.f, 1.5f, 2);  // 玻璃
        addSphere(1.f, -4.f, 1.f, 0.f,  0.4f, 0.2f, 0.1f, 0.f, 0); // Lambert 棕
        addSphere(1.f,  4.f, 1.f, 0.f,  0.7f, 0.6f, 0.5f, 0.f, 1); // Metal 金

        // 随机小球（RTIOW 风格，11x11 网格）
        int seed = 0;
        for (int a = -11; a < 11; ++a) {
            for (int b = -11; b < 11; ++b) {
                float cx = a + 0.9f * ihash(seed++);
                float cy = 0.2f;
                float cz = b + 0.9f * ihash(seed++);
                // 跳过与三个特写球重叠的位置
                auto d = [](float x, float z, float ox, float oz) {
                    float dx = x - ox, dz = z - oz;
                    return std::sqrt(dx*dx + dz*dz);
                };
                if (d(cx, cz, 0.f, 0.f) < 1.3f) continue;
                if (d(cx, cz,-4.f, 0.f) < 1.3f) continue;
                if (d(cx, cz, 4.f, 0.f) < 1.3f) continue;

                float choose = ihash(seed++);
                if (choose < 0.7f) { // 漫反射
                    float r1 = ihash(seed++); float r2 = ihash(seed++);
                    float g1 = ihash(seed++); float g2 = ihash(seed++);
                    float b1 = ihash(seed++); float b2 = ihash(seed++);
                    addSphere(0.2f, cx, cy, cz, r1*r2, g1*g2, b1*b2, 0.f, 0);
                } else if (choose < 0.9f) { // 金属
                    float r = 0.5f + 0.5f * ihash(seed++);
                    float g = 0.5f + 0.5f * ihash(seed++);
                    float b2= 0.5f + 0.5f * ihash(seed++);
                    float fuzz = 0.5f * ihash(seed++);
                    addSphere(0.2f, cx, cy, cz, r, g, b2, fuzz, 1);
                } else { // 玻璃
                    addSphere(0.2f, cx, cy, cz, 1.f, 1.f, 1.f, 1.5f, 2);
                }
            }
        }

        uint32_t numInstances = static_cast<uint32_t>(instances.size());

        // 实例数据缓冲
        VkBuffer instanceBuf;
        VkDeviceMemory instanceMem;
        VkDeviceSize instBytes = numInstances * sizeof(VkAccelerationStructureInstanceKHR);
        createBufferForAS(instBytes,
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                          instances.data(), instBytes, instanceBuf, instanceMem);

        VkAccelerationStructureGeometryKHR tlasGeom{};
        tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
        tlasGeom.geometry.instances.data.deviceAddress = getBufferDeviceAddress(instanceBuf);

        VkAccelerationStructureBuildGeometryInfoKHR tlasInfo{};
        tlasInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        tlasInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tlasInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        tlasInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tlasInfo.geometryCount = 1;
        tlasInfo.pGeometries = &tlasGeom;

        fpGetAccelBuildSizes_(
            device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasInfo, &numInstances, &sizeInfo);

        createBuffer(sizeInfo.accelerationStructureSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     tlasBuffer_,
                     tlasMemory_);

        VkAccelerationStructureCreateInfoKHR tlasCI{};
        tlasCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        tlasCI.buffer = tlasBuffer_;
        tlasCI.size = sizeInfo.accelerationStructureSize;
        tlasCI.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        VK_CHECK(fpCreateAccelStructure_(device_, &tlasCI, nullptr, &tlas_));

        VkBuffer scratchBuf2;
        VkDeviceMemory scratchMem2;
        createBuffer(sizeInfo.buildScratchSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     scratchBuf2,
                     scratchMem2);

        tlasInfo.dstAccelerationStructure = tlas_;
        tlasInfo.scratchData.deviceAddress = getBufferDeviceAddress(scratchBuf2);

        VkAccelerationStructureBuildRangeInfoKHR tlasRange{numInstances, 0, 0, 0};
        const VkAccelerationStructureBuildRangeInfoKHR* pTlasRange = &tlasRange;

        cmd = beginSingleTimeCommands();
        fpCmdBuildAccelStructures_(cmd, 1, &tlasInfo, &pTlasRange);
        endSingleTimeCommands(cmd);

        // ── 材质 SSBO ────────────────────────────────────────────────────
        VkDeviceSize matBytes = materials.size() * sizeof(MaterialGPU);
        createBufferForAS(matBytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                          materials.data(), matBytes, materialBuffer_, materialMemory_);

        // 清理临时缓冲
        vkDestroyBuffer(device_, vertBuf, nullptr);
        vkFreeMemory(device_, vertMem, nullptr);
        vkDestroyBuffer(device_, idxBuf, nullptr);
        vkFreeMemory(device_, idxMem, nullptr);
        vkDestroyBuffer(device_, scratchBuf, nullptr);
        vkFreeMemory(device_, scratchMem, nullptr);
        vkDestroyBuffer(device_, instanceBuf, nullptr);
        vkFreeMemory(device_, instanceMem, nullptr);
        vkDestroyBuffer(device_, scratchBuf2, nullptr);
        vkFreeMemory(device_, scratchMem2, nullptr);

        std::cout << "✅ 加速结构构建完成（BLAS: 单位球体，TLAS: " << numInstances << " 个球体实例）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 创建光线追踪管线
    // ═══════════════════════════════════════════════════════════════════════

    void createRTPipeline() {
        // ── 描述符集布局：binding 0=TLAS, 1=存储图像, 2=材质SSBO, 3=累积图像 ─
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        bindings[2].binding         = 2;
        bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        bindings[3].binding         = 3;
        bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

        VkDescriptorSetLayoutCreateInfo dslCI{};
        dslCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslCI.bindingCount = static_cast<uint32_t>(bindings.size());
        dslCI.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dslCI, nullptr, &rtDescSetLayout_));

        // ── 描述符池 ──────────────────────────────────────────────────────
        std::array<VkDescriptorPoolSize, 3> poolSizes{};
        poolSizes[0] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1};
        poolSizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2};  // outputImage + accumImage
        poolSizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.maxSets       = 1;
        poolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolCI.pPoolSizes    = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(device_, &poolCI, nullptr, &rtDescPool_));

        VkDescriptorSetAllocateInfo dsAI{};
        dsAI.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAI.descriptorPool     = rtDescPool_;
        dsAI.descriptorSetCount = 1;
        dsAI.pSetLayouts        = &rtDescSetLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &dsAI, &rtDescSet_));

        // ── 写入描述符：TLAS ──────────────────────────────────────────────
        VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
        asWrite.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        asWrite.accelerationStructureCount = 1;
        asWrite.pAccelerationStructures    = &tlas_;

        VkWriteDescriptorSet wTlas{};
        wTlas.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wTlas.pNext           = &asWrite;
        wTlas.dstSet          = rtDescSet_;
        wTlas.dstBinding      = 0;
        wTlas.descriptorCount = 1;
        wTlas.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

        // ── 写入描述符：存储图像 ──────────────────────────────────────────
        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView   = rtOutputView_;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet wImg{};
        wImg.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wImg.dstSet          = rtDescSet_;
        wImg.dstBinding      = 1;
        wImg.descriptorCount = 1;
        wImg.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wImg.pImageInfo      = &imgInfo;

        // ── 写入描述符：材质 SSBO ─────────────────────────────────────────
        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = materialBuffer_;
        bufInfo.offset = 0;
        bufInfo.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet wMat{};
        wMat.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wMat.dstSet          = rtDescSet_;
        wMat.dstBinding      = 2;
        wMat.descriptorCount = 1;
        wMat.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wMat.pBufferInfo     = &bufInfo;

        // ── 写入描述符：累积图像 ──────────────────────────────────────────
        VkDescriptorImageInfo accumImgInfo{};
        accumImgInfo.imageView   = accumView_;
        accumImgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet wAccum{};
        wAccum.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wAccum.dstSet          = rtDescSet_;
        wAccum.dstBinding      = 3;
        wAccum.descriptorCount = 1;
        wAccum.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wAccum.pImageInfo      = &accumImgInfo;

        std::array<VkWriteDescriptorSet, 4> writes = {wTlas, wImg, wMat, wAccum};
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        // ── 管线布局（含 push constant：frameAccum） ──────────────────────
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(uint32_t);

        VkPipelineLayoutCreateInfo pli{};
        pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount         = 1;
        pli.pSetLayouts            = &rtDescSetLayout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges    = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &rtPipelineLayout_));

        // ── 加载着色器模块 ────────────────────────────────────────────────
        VkShaderModule rgenMod  = createShaderModuleFromFile(device_, "ray_gen.rgen.spv");
        VkShaderModule missMod  = createShaderModuleFromFile(device_, "miss.rmiss.spv");
        VkShaderModule rchitMod = createShaderModuleFromFile(device_, "closest_hit.rchit.spv");

        // ── 着色器阶段（顺序与 Shader Group 索引对应） ────────────────────
        std::array<VkPipelineShaderStageCreateInfo, 3> stages{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr, 0, VK_SHADER_STAGE_RAYGEN_BIT_KHR,  rgenMod,  "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr, 0, VK_SHADER_STAGE_MISS_BIT_KHR,    missMod,  "main", nullptr};
        stages[2] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr, 0, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, rchitMod, "main", nullptr};

        // ── Shader Groups（SBT 槽） ────────────────────────────────────────
        // Group 0 → rgen  (GENERAL)
        // Group 1 → rmiss (GENERAL)
        // Group 2 → rchit (TRIANGLES_HIT_GROUP)
        std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> groups{};
        for (auto& g : groups) {
            g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            g.generalShader      = VK_SHADER_UNUSED_KHR;
            g.closestHitShader   = VK_SHADER_UNUSED_KHR;
            g.anyHitShader       = VK_SHADER_UNUSED_KHR;
            g.intersectionShader = VK_SHADER_UNUSED_KHR;
        }
        groups[0].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[0].generalShader = 0;
        groups[1].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[1].generalShader = 1;
        groups[2].type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        groups[2].closestHitShader = 2;

        // ── 创建光线追踪管线 ──────────────────────────────────────────────
        VkRayTracingPipelineCreateInfoKHR rtpCI{};
        rtpCI.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        rtpCI.stageCount                   = static_cast<uint32_t>(stages.size());
        rtpCI.pStages                      = stages.data();
        rtpCI.groupCount                   = static_cast<uint32_t>(groups.size());
        rtpCI.pGroups                      = groups.data();
        rtpCI.maxPipelineRayRecursionDepth = 1;
        rtpCI.layout                       = rtPipelineLayout_;

        VK_CHECK(fpCreateRTPipelines_(device_, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtpCI, nullptr, &rtPipeline_));

        vkDestroyShaderModule(device_, rgenMod,  nullptr);
        vkDestroyShaderModule(device_, missMod,  nullptr);
        vkDestroyShaderModule(device_, rchitMod, nullptr);

        std::cout << "✅ 光线追踪管线已创建（rgen + miss + closest_hit）\n";
    }

    void createShaderBindingTable() {
        // 查询光线追踪管线属性（句柄大小、对齐要求）
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
        rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &rtProps;
        vkGetPhysicalDeviceProperties2(physicalDevice_, &props2);

        const uint32_t handleSize      = rtProps.shaderGroupHandleSize;
        const uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
        // 每条 SBT 记录大小需对齐到 handleAlignment
        const uint32_t stride = (handleSize + handleAlignment - 1) & ~(handleAlignment - 1);
        // SBT 基地址需对齐到 shaderGroupBaseAlignment（通常 64 字节）
        const uint32_t baseAlign = rtProps.shaderGroupBaseAlignment;

        const uint32_t groupCount = 3; // rgen + miss + hit
        const uint32_t dataSize   = groupCount * handleSize;
        std::vector<uint8_t> handles(dataSize);
        VK_CHECK(fpGetRTShaderGroupHandles_(device_, rtPipeline_, 0, groupCount, dataSize, handles.data()));

        // 每个区域各占一个 stride，基地址按 baseAlign 对齐排列
        const VkDeviceSize rgenOffset = 0;
        const VkDeviceSize missOffset = (stride + baseAlign - 1) & ~(VkDeviceSize)(baseAlign - 1);
        const VkDeviceSize hitOffset  = missOffset + ((stride + baseAlign - 1) & ~(VkDeviceSize)(baseAlign - 1));
        const VkDeviceSize sbtSize    = hitOffset + stride;

        createBuffer(sbtSize,
                     VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     sbtBuffer_,
                     sbtMemory_);

        uint8_t* mapped = nullptr;
        VK_CHECK(vkMapMemory(device_, sbtMemory_, 0, sbtSize, 0, reinterpret_cast<void**>(&mapped)));
        std::memcpy(mapped + rgenOffset, handles.data() + 0 * handleSize, handleSize);
        std::memcpy(mapped + missOffset, handles.data() + 1 * handleSize, handleSize);
        std::memcpy(mapped + hitOffset,  handles.data() + 2 * handleSize, handleSize);
        vkUnmapMemory(device_, sbtMemory_);

        const VkDeviceAddress sbtAddr = getBufferDeviceAddress(sbtBuffer_);
        sbtRgen_ = {sbtAddr + rgenOffset, stride, stride};
        sbtMiss_ = {sbtAddr + missOffset, stride, stride};
        sbtHit_  = {sbtAddr + hitOffset,  stride, stride};
        sbtCall_ = {};

        std::cout << "✅ Shader Binding Table 已创建（handleSize=" << handleSize << "B, stride=" << stride << "B）\n";
    }

    void createAccumImage() {
        VkImageCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType     = VK_IMAGE_TYPE_2D;
        ci.extent        = {swapchainExtent_.width, swapchainExtent_.height, 1};
        ci.mipLevels     = 1;
        ci.arrayLayers   = 1;
        ci.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
        ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        VK_CHECK(vkCreateImage(device_, &ci, nullptr, &accumImage_));

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device_, accumImage_, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &accumMemory_));
        VK_CHECK(vkBindImageMemory(device_, accumImage_, accumMemory_, 0));

        VkImageViewCreateInfo vci{};
        vci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image            = accumImage_;
        vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        vci.format           = VK_FORMAT_R32G32B32A32_SFLOAT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &accumView_));

        // 初始化布局：UNDEFINED → GENERAL
        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.image               = accumImage_;
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.srcAccessMask       = 0;
        b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        endSingleTimeCommands(cmd);
    }

    void createRTOutputImage() {
        // 光线追踪写入存储图像（VK_IMAGE_USAGE_STORAGE_BIT）
        // 渲染完成后 blit 到交换链图像显示
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.extent = {swapchainExtent_.width, swapchainExtent_.height, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.format = VK_FORMAT_B8G8R8A8_UNORM;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage = VK_IMAGE_USAGE_STORAGE_BIT |     // 光线追踪着色器写入
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT; // blit 到交换链
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        VK_CHECK(vkCreateImage(device_, &ci, nullptr, &rtOutputImage_));

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device_, rtOutputImage_, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &rtOutputMemory_));
        VK_CHECK(vkBindImageMemory(device_, rtOutputImage_, rtOutputMemory_, 0));

        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = rtOutputImage_;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_B8G8R8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &rtOutputView_));

        std::cout << "✅ 光线追踪输出图像已创建（Storage Image，blit 到交换链）\n";
    }

    void recordRTCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        // 布局转换：UNDEFINED → GENERAL（Storage Image 写入前需要的布局）
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.image = rtOutputImage_;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &barrier);

        // ── 绑定光线追踪管线 ──────────────────────────────────────────────
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                rtPipelineLayout_, 0, 1, &rtDescSet_, 0, nullptr);
        vkCmdPushConstants(cmd, rtPipelineLayout_, VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                           0, sizeof(uint32_t), &accumFrameCount_);

        // ── 发射光线！ ─────────────────────────────────────────────────────
        // vkCmdTraceRaysKHR 替代 vkCmdDraw
        // 参数：SBT 各部分的 DeviceAddress + 像素尺寸

        if (fpCmdTraceRays_) {
            fpCmdTraceRays_(cmd,
                            &sbtRgen_,
                            &sbtMiss_,
                            &sbtHit_,
                            &sbtCall_,
                            swapchainExtent_.width,  // 水平像素数
                            swapchainExtent_.height, // 垂直像素数
                            1);                      // depth（通常为1）
        }

        // ── Blit 结果到交换链图像 ─────────────────────────────────────────
        // 布局转换：GENERAL → TRANSFER_SRC
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &barrier);

        // 交换链图像：UNDEFINED → TRANSFER_DST
        VkImageMemoryBarrier swapBarrier{};
        swapBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        swapBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        swapBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapBarrier.image = swapchainImages_[imageIndex];
        swapBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        swapBarrier.srcAccessMask = 0;
        swapBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        swapBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        swapBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &swapBarrier);

        VkImageBlit blitRegion{};
        blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blitRegion.srcOffsets[1] = {(int32_t)swapchainExtent_.width, (int32_t)swapchainExtent_.height, 1};
        blitRegion.dstSubresource = blitRegion.srcSubresource;
        blitRegion.dstOffsets[1] = blitRegion.srcOffsets[1];
        vkCmdBlitImage(cmd,
                       rtOutputImage_,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       swapchainImages_[imageIndex],
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &blitRegion,
                       VK_FILTER_NEAREST);

        // 交换链图像：TRANSFER_DST → PRESENT_SRC
        swapBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        swapBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        swapBarrier.dstAccessMask = 0;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &swapBarrier);

        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t imgIdx = 0;
        VkResult r = vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX, imageAvailableSems_[currentFrame_], VK_NULL_HANDLE, &imgIdx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            return;
        }
        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordRTCommandBuffer(commandBuffers_[currentFrame_], imgIdx);
        VkSemaphore ws[] = {imageAvailableSems_[currentFrame_]};
        VkPipelineStageFlags wst[] = {VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR};
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
        vkQueuePresentKHR(presentQueue_, &pi);
        currentFrame_ = (currentFrame_ + 1) % 2;
        ++accumFrameCount_;
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    void printRayTracingGuide() {
        std::cout << "\n════════════════════════════════════════════\n";
        std::cout << " Vulkan 光线追踪 API 概念速查\n";
        std::cout << "════════════════════════════════════════════\n\n";
        std::cout << "1. 所需扩展：\n";
        std::cout << "   VK_KHR_ray_tracing_pipeline\n";
        std::cout << "   VK_KHR_acceleration_structure\n";
        std::cout << "   VK_KHR_buffer_device_address\n\n";
        std::cout << "2. 加速结构（AS）：\n";
        std::cout << "   BLAS = 单个几何体的三角形网格\n";
        std::cout << "   TLAS = 场景中 BLAS 实例的集合\n\n";
        std::cout << "3. 新着色器类型：\n";
        std::cout << "   .rgen  → 光线生成（每像素调用）\n";
        std::cout << "   .rmiss → 未命中（返回天空颜色）\n";
        std::cout << "   .rchit → 最近命中（计算光照）\n";
        std::cout << "   .rahit → 任意命中（透明/Alpha Test）\n\n";
        std::cout << "4. 渲染命令：\n";
        std::cout << "   vkCmdTraceRaysKHR(width, height, 1)\n";
        std::cout << "   替代 vkCmdDraw\n\n";
        std::cout << "5. GLSL 光线追踪着色器语法：\n";
        std::cout << "   #extension GL_EXT_ray_tracing : require\n";
        std::cout << "   traceRayEXT(tlas, flags, mask, sbtOffset, sbtStride,\n";
        std::cout << "               missIndex, origin, tMin, dir, tMax, payload)\n\n";
        std::cout << "6. 递归光线追踪（反射/折射）：\n";
        std::cout << "   在 .rchit 中再次调用 traceRayEXT()\n";
        std::cout << "   最大递归深度受 maxRayRecursionDepth 限制\n";
    }

    // ─── 辅助函数 ─────────────────────────────────────────────────────────────

    uint32_t findMemoryType(uint32_t f, VkMemoryPropertyFlags p) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((f & (1u << i)) && (mp.memoryTypes[i].propertyFlags & p) == p)
                return i;
        throw std::runtime_error("找不到内存类型");
    }

    void createBuffer(VkDeviceSize sz, VkBufferUsageFlags u, VkMemoryPropertyFlags p, VkBuffer& b, VkDeviceMemory& m) {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = sz;
        ci.usage = u;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &ci, nullptr, &b));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(device_, b, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, p);
        // Buffer Device Address 需要 DEVICE_ADDRESS 标志
        VkMemoryAllocateFlagsInfo flagsInfo{};
        if (u & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
            flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
            flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            ai.pNext = &flagsInfo;
        }
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &m));
        VK_CHECK(vkBindBufferMemory(device_, b, m, 0));
    }

    void createBufferForAS(VkDeviceSize sz,
                           VkBufferUsageFlags usage,
                           const void* data,
                           size_t dataSize,
                           VkBuffer& buf,
                           VkDeviceMemory& mem) {
        createBuffer(sz, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buf, mem);
        if (data) {
            void* mapped = nullptr;
            vkMapMemory(device_, mem, 0, dataSize, 0, &mapped);
            std::memcpy(mapped, data, dataSize);
            vkUnmapMemory(device_, mem);
        }
    }

    VkDeviceAddress getBufferDeviceAddress(VkBuffer buf) {
        VkBufferDeviceAddressInfo info{};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        info.buffer = buf;
        return vkGetBufferDeviceAddress(device_, &info);
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandPool = commandPool_;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device_, &ai, &cmd);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        return cmd;
    }

    void endSingleTimeCommands(VkCommandBuffer cmd) {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue_);
        vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
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
#ifdef __APPLE__
        ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    }
    void createSurface() {
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
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
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT; // blit 目标
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
    void cleanup() {
        if (blas_ != VK_NULL_HANDLE)
            fpDestroyAccelStructure_(device_, blas_, nullptr);
        if (tlas_ != VK_NULL_HANDLE)
            fpDestroyAccelStructure_(device_, tlas_, nullptr);
        if (blasBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, blasBuffer_, nullptr);
            vkFreeMemory(device_, blasMemory_, nullptr);
        }
        if (tlasBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, tlasBuffer_, nullptr);
            vkFreeMemory(device_, tlasMemory_, nullptr);
        }
        if (rtOutputView_ != VK_NULL_HANDLE)
            vkDestroyImageView(device_, rtOutputView_, nullptr);
        if (rtOutputImage_ != VK_NULL_HANDLE) {
            vkDestroyImage(device_, rtOutputImage_, nullptr);
            vkFreeMemory(device_, rtOutputMemory_, nullptr);
        }
        if (accumView_ != VK_NULL_HANDLE)
            vkDestroyImageView(device_, accumView_, nullptr);
        if (accumImage_ != VK_NULL_HANDLE) {
            vkDestroyImage(device_, accumImage_, nullptr);
            vkFreeMemory(device_, accumMemory_, nullptr);
        }
        if (sbtBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, sbtBuffer_, nullptr);
            vkFreeMemory(device_, sbtMemory_, nullptr);
        }
        if (materialBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, materialBuffer_, nullptr);
            vkFreeMemory(device_, materialMemory_, nullptr);
        }
        if (rtPipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, rtPipeline_, nullptr);
        if (rtPipelineLayout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, rtPipelineLayout_, nullptr);
        if (rtDescPool_ != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device_, rtDescPool_, nullptr);
        if (rtDescSetLayout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, rtDescSetLayout_, nullptr);
        for (int i = 0; i < 2; ++i) {
            if (i < (int)imageAvailableSems_.size()) {
                vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
                vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
                vkDestroyFence(device_, inFlightFences_[i], nullptr);
            }
        }
        if (commandPool_)
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        if (swapchain_)
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        if (device_)
            vkDestroyDevice(device_, nullptr);
        if (surface_)
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
        if (instance_)
            vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
};

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════"
                 "════\n";
    std::cout << " 第20章：硬件光线追踪（VK_KHR_ray_tracing_pipeline）\n";
    std::cout << "\n";
    std::cout << " ⚠️  需要 NVIDIA RTX / AMD RDNA2+ GPU\n";
    std::cout << " ⚠️  macOS/MoltenVK 暂不支持（Metal 光线追踪 API "
                 "不同）\n";
    std::cout << "═══════════════════════════════════════════════════════════════"
                 "════\n\n";
    Ch20App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
    return 0;
}
