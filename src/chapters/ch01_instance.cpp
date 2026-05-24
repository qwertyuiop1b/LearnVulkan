/**
 * @file ch01_instance.cpp
 * @brief 第01章：创建 VkInstance 与验证层
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【核心概念】
 *
 * VkInstance（Vulkan 实例）是整个 Vulkan 应用程序的根对象。
 * 它代表了应用程序与 Vulkan 运行时库之间的连接。
 *
 * 【创建流程】
 *
 *  ┌─────────────────────────────────────────────────┐
 *  │  VkApplicationInfo  ─→  描述应用程序信息         │
 *  │  VkInstanceCreateInfo ─→  告诉 Vulkan 要哪些      │
 *  │      extensions（扩展）和 layers（层）            │
 *  │  vkCreateInstance() ─→  创建实例                 │
 *  └─────────────────────────────────────────────────┘
 *
 * 【验证层（Validation Layers）】
 *
 * Vulkan 本身不做错误检查（为了性能），验证层是可选插入的调试组件：
 *  - 检查 API 调用参数是否合法
 *  - 追踪对象的创建与销毁
 *  - 检查线程安全
 *  - 记录每次 API 调用
 *  - 追踪 Vulkan 调用以进行性能分析
 *
 * 生产构建时（NDEBUG）完全禁用，开销为零。
 *
 * 【调试信使（Debug Messenger）】
 *
 * VK_EXT_debug_utils 扩展允许我们注册回调函数接收验证层消息。
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan/vulkan_core.h>
#include <vulkan_tutorial/utils.hpp>
#include <iostream>
#include <stdexcept>

// ─── vkCreateDebugUtilsMessengerEXT 不在 Vulkan 核心中 ──────────────────────
// 需要通过 vkGetInstanceProcAddr 动态加载

static VkResult createDebugUtilsMessengerEXT(
    VkInstance                                instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks*              pAllocator,
    VkDebugUtilsMessengerEXT*                 pMessenger)
{
    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (!func) return VK_ERROR_EXTENSION_NOT_PRESENT;
    return func(instance, pCreateInfo, pAllocator, pMessenger);
}

static void destroyDebugUtilsMessengerEXT(
    VkInstance                   instance,
    VkDebugUtilsMessengerEXT     messenger,
    const VkAllocationCallbacks* pAllocator)
{
    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func) func(instance, messenger, pAllocator);
}

// ─── 应用程序类 ────────────────────────────────────────────────────────────────

class Ch01App {
public:
    void run()
    {
        createInstance();
        setupDebugMessenger();
        printInstanceInfo();
        cleanup();
    }

private:
    VkInstance               instance_  = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMsgr_ = VK_NULL_HANDLE;

    // ── Step 1: 创建 VkInstance ───────────────────────────────────────────────

    void createInstance()
    {
        // 验证层检查
        if (ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport())
            throw std::runtime_error(
                "请求的验证层不可用！请安装 Vulkan SDK。");

        // ① VkApplicationInfo：告知驱动程序我们的应用程序信息
        //   驱动程序可利用此信息进行针对性优化（例如识别已知引擎）
        VkApplicationInfo appInfo{};
        appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName   = "Ch01 - Vulkan Instance";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName        = "No Engine";
        appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion         = VK_API_VERSION_1_4;

        // ② 获取所需扩展（GLFW 需要的窗口扩展 + 调试扩展 + macOS 扩展）
        auto extensions = getRequiredInstanceExtensions();

        // ③ VkInstanceCreateInfo：实例创建的主配置结构体
        VkInstanceCreateInfo createInfo{};
        createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo        = &appInfo;
        createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // macOS + MoltenVK 必须设置此标志，允许枚举非完全符合规范的物理设备
#ifdef __APPLE__
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        // ④ 配置验证层
        //   注意：在 pNext 中传入 debugCreateInfo，可以捕获
        //   vkCreateInstance/vkDestroyInstance 本身的调试消息
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (ENABLE_VALIDATION_LAYERS) {
            createInfo.enabledLayerCount   = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
        }

        // ⑤ 创建实例！
        VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance_));

        std::cout << "✅ VkInstance 创建成功！\n";
    }

    // ── Step 2: 注册调试信使 ──────────────────────────────────────────────────

    void setupDebugMessenger()
    {
        if (!ENABLE_VALIDATION_LAYERS) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);

        VK_CHECK(createDebugUtilsMessengerEXT(
            instance_, &createInfo, nullptr, &debugMsgr_));

        std::cout << "✅ 调试信使已注册！\n";
    }

    // ── 打印实例信息 ──────────────────────────────────────────────────────────

    void printInstanceInfo()
    {
        // 枚举所有可用扩展
        uint32_t extCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, exts.data());

        std::cout << "\n📋 系统支持的 Instance 扩展（共 " << extCount << " 个）：\n";
        for (const auto& e : exts)
            std::cout << "  - " << e.extensionName
                      << " (spec version " << e.specVersion << ")\n";

        // 枚举所有可用验证层
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

        std::cout << "\n🛡️  系统支持的验证层（共 " << layerCount << " 个）：\n";
        for (const auto& l : layers)
            std::cout << "  - " << l.layerName << "\n"
                      << "    " << l.description << "\n";
    }

    // ── 清理资源（RAII 思想：创建与销毁成对出现）────────────────────────────

    void cleanup()
    {
        // 销毁顺序与创建顺序相反！
        if (ENABLE_VALIDATION_LAYERS && debugMsgr_ != VK_NULL_HANDLE)
            destroyDebugUtilsMessengerEXT(instance_, debugMsgr_, nullptr);

        if (instance_ != VK_NULL_HANDLE)
            vkDestroyInstance(instance_, nullptr);

        std::cout << "\n✅ 资源已清理完毕。\n";
    }
};

int main()
{
    std::cout << "═══════════════════════════════════════\n";
    std::cout << " 第01章：VkInstance 与验证层\n";
    std::cout << "═══════════════════════════════════════\n\n";

    Ch01App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
