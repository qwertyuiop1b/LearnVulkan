# vk-bootstrap 到 Vulkan-Hpp RAII 的异常安全所有权交接

状态：设计记录，尚未应用到 `src/vk_context.cpp`。

## 目标

`vk-bootstrap` 负责简化实例、物理设备和逻辑设备的创建与选择；`vk::raii` 负责项目内 Vulkan 句柄的长期生命周期。两者可以一起使用，但同一个 Vulkan 句柄在任意时刻只能有一个销毁者。

当前 `VkContext` 在构造函数末尾才将 `vkb::Instance` 和 `vkb::Device` 的原始句柄置空。若在这之前的表面创建、物理设备选择或逻辑设备创建步骤抛出异常，`vkb::Instance` 与 `vkb::Device` 都只是普通结构体，不会自动调用销毁函数，因此会泄漏已创建的资源。

设计目标是在每个句柄创建成功后立即安装清理动作；仅在 RAII 包装成功接管后取消该动作。这样异常路径由 `ScopeExit` 清理，成功路径只由 `vk::raii` 销毁，不会 double free。

## 创建期调试消息

`vk::raii::DebugUtilsMessengerEXT` 只能在 `vk::raii::Instance` 已存在后构造，因此它无法接收 `vkCreateInstance` 过程中产生的 validation 消息。

保留 `vkb::InstanceBuilder::set_debug_callback(DebugCallback)`。vk-bootstrap 会在创建实例时把调试 messenger 的创建信息链入实例创建流程，从而覆盖实例创建期的消息。随后将 vk-bootstrap 创建的 `debug_messenger` 交给 `vk::raii::DebugUtilsMessengerEXT`，而不是再创建一个新的 messenger。

## ScopeExit 实现

将该工具放在项目的通用工具头中，例如 `src/vk_utils.h`。回调必须是 `noexcept`：析构函数期间不能让异常逃逸。

```cpp
#include <concepts>
#include <type_traits>
#include <utility>

template <typename F>
    requires std::is_nothrow_invocable_v<F&> && std::is_nothrow_move_constructible_v<F>
class ScopeExit
{
public:
    explicit ScopeExit(F callback) noexcept
        : callback_(std::move(callback))
    {
    }

    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

    ScopeExit(ScopeExit&& other) noexcept
        : callback_(std::move(other.callback_))
        , active_(std::exchange(other.active_, false))
    {
    }

    ~ScopeExit() noexcept
    {
        if (active_)
        {
            callback_();
        }
    }

    void Release() noexcept
    {
        active_ = false;
    }

private:
    F callback_;
    bool active_ = true;
};

template <typename F>
ScopeExit(F) -> ScopeExit<F>;
```

`Release()` 只在另一方已成功接管句柄后调用。该类型不可复制，移动时通过 `std::exchange` 让源对象失效，因此一个清理动作最多执行一次。

下文示例中的 `allocator` 指 Vulkan-Hpp RAII 构造函数使用的 allocator 参数，`nativeAllocator` 指 C API、GLFW 和 vk-bootstrap 使用的 `VkAllocationCallbacks*`。当前项目没有自定义 allocator，两个值均为 `nullptr`。引入自定义 allocator 时，两者必须指向同一份 callbacks 数据，不能临时构造或复制其中一份。

## Instance 与 debug messenger 的交接

`vkb::destroy_instance` 会先销毁 `debug_messenger`，再销毁实例。因此，实例的 vk-bootstrap 清理守卫在实例转交后必须拆分成两个守卫：RAII 管理实例，新的 `debugCleanup` 管理尚未转交的 debug messenger。

```cpp
vkb::Instance vkbInstance = vkbInstanceRet.value();
auto instanceCleanup = ScopeExit{[&]() noexcept {
    vkb::destroy_instance(vkbInstance);
}};

vk::raii::Instance localInstance(context, vkbInstance.instance, allocator);

const VkInstance rawInstance = vkbInstance.instance;
auto debugCleanup = ScopeExit{[&]() noexcept {
    if (vkbInstance.debug_messenger != VK_NULL_HANDLE)
    {
        vkb::destroy_debug_utils_messenger(
            rawInstance,
            vkbInstance.debug_messenger,
            vkbInstance.allocation_callbacks);
    }
}};

// 以下两句不抛异常。此后实例由 localInstance 销毁。
vkbInstance.instance = VK_NULL_HANDLE;
instanceCleanup.Release();

vk::raii::DebugUtilsMessengerEXT localDebugMessenger(
    localInstance, vkbInstance.debug_messenger, allocator);

// 以下两句不抛异常。此后 messenger 由 localDebugMessenger 销毁。
vkbInstance.debug_messenger = VK_NULL_HANDLE;
debugCleanup.Release();
```

若 `localInstance` 构造失败，`instanceCleanup` 同时清理 vk-bootstrap 的 messenger 和实例。若 `localDebugMessenger` 构造失败，`debugCleanup` 清理 messenger，而局部 `localInstance` 在栈展开时清理实例。

确认两个局部 RAII 对象都已接管后，再以 move assignment 写入 `VkContext` 成员：

```cpp
instance = std::move(localInstance);
debugMessenger = std::move(localDebugMessenger);
```

成员对象已经负责析构，因此之后的 surface、physical-device 选择或 device 创建失败时，已交接的实例和 messenger 也会自动释放。

## Surface 的交接

GLFW 创建 surface 后，RAII 包装前也存在一个异常窗口。使用 Vulkan 销毁函数作为守卫：

```cpp
VkSurfaceKHR rawSurface = VK_NULL_HANDLE;
VK_CHECK(glfwCreateWindowSurface(rawInstance, &vkWindow.GetWindow(), nativeAllocator, &rawSurface));

auto surfaceCleanup = ScopeExit{[&]() noexcept {
    vkDestroySurfaceKHR(rawInstance, rawSurface, nativeAllocator);
}};

vk::raii::SurfaceKHR localSurface(instance, rawSurface, allocator);
surfaceCleanup.Release();
surface = std::move(localSurface);
```

`vk::raii::PhysicalDevice` 只是非拥有包装，不调用 `vkDestroyPhysicalDevice`，不需要 ScopeExit。

## Device 的交接

逻辑设备由 vk-bootstrap 创建后，使用 `vkb::destroy_device` 保护 RAII 包装前的窗口：

```cpp
vkb::Device vkbDevice = deviceRet.value();
auto deviceCleanup = ScopeExit{[&]() noexcept {
    vkb::destroy_device(vkbDevice);
}};

vk::raii::Device localDevice(physicalDevice, vkbDevice.device, allocator);

// 以下两句不抛异常。此后逻辑设备由 localDevice 销毁。
vkbDevice.device = VK_NULL_HANDLE;
deviceCleanup.Release();

device = std::move(localDevice);
```

不要在成功交接后调用 `vkb::destroy_instance` 或 `vkb::destroy_device`；它们会与 RAII 析构竞争同一个句柄。

## 自定义 allocator

目前代码全部使用 `nullptr` allocator，因此创建和销毁是一致的。未来使用 `VkAllocationCallbacks` 时，必须让同一个 allocator 贯穿以下所有操作：

1. 传给 `vkb::InstanceBuilder::set_allocation_callbacks` 和 `vkb::DeviceBuilder::set_allocation_callbacks`。
2. 传给 `glfwCreateWindowSurface`，以及 ScopeExit 中的 `vkDestroySurfaceKHR`。
3. 传给 `vk::raii::Instance`、`DebugUtilsMessengerEXT`、`SurfaceKHR` 和 `Device` 的构造函数。
4. 让 allocator 对象及其 `pUserData` 的寿命长于全部 RAII 成员。若它是 `VkContext` 成员，必须声明在这些 RAII 成员之前，保证它最后销毁。

不能在 vk-bootstrap 中使用自定义 allocator，却向 RAII 构造函数传 `nullptr`；对应 Vulkan 销毁调用会收到不一致的 allocator。

## 实施顺序与验证

1. 在 `vk_utils.h` 加入并单测 `ScopeExit`：正常离开执行一次、`Release()` 后不执行、移动后仅目标执行。
2. 先改 instance/debug messenger 的交接，再改 surface，最后改 device。
3. 在实例、surface、选择物理设备和逻辑设备创建之后分别注入一次异常，运行泄漏检测工具确认每条路径均释放资源。
4. 保持 `VkContext` 的成员声明顺序：`Device` 最先析构，随后 `PhysicalDevice`、`Surface`、`DebugUtilsMessenger`、`Instance`。当前声明顺序已经满足此要求。
