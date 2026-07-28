#include "vk_context.h"
#include <stdexcept>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_core.h>
#include <VkBootstrap.h>

#include "vk_window.h"
#include "vk_utils.h"

namespace vk_engine
{
VkContext::VkContext(const VkWindow& inWindow) : vkWindow(inWindow)
{
    vkb::InstanceBuilder builder;
    const auto vkbInstanceRet = builder.set_app_name("vk-engine")
                                    .set_app_version(1, 0, 0)
                                    .set_engine_name("No Engine")
                                    .set_engine_version(1, 0, 0)
                                    .require_api_version(1, 3, 0)
                                    .enable_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
                                    .request_validation_layers()
                                    .set_debug_callback(DebugCallback)
                                    .build();
    if (!vkbInstanceRet)
    {
        throw std::runtime_error("failed to create instance");
    }

    // instance
    vkb::Instance vkbInstance = vkbInstanceRet.value();
    vk::raii::Context context;
    instance = vk::raii::Instance(context, vkbInstance.instance);

    // debug messenger
    debugMessenger = vk::raii::DebugUtilsMessengerEXT(instance, vkbInstance.debug_messenger);

    // surface
    VkSurfaceKHR rawSurface;
    VK_CHECK(glfwCreateWindowSurface(vkbInstance.instance, &vkWindow.GetWindow(), nullptr, &rawSurface));
    surface = vk::raii::SurfaceKHR(instance, rawSurface);

    vkb::PhysicalDeviceSelector selector{vkbInstance};
    const auto phyRet = selector.set_surface(*surface).select();
    if (!phyRet)
    {
        throw std::runtime_error("failed to select physical");
    }

    // physical device
    vkb::PhysicalDevice vkbPhysicalDevice = phyRet.value();
    physicalDevice = vk::raii::PhysicalDevice(instance, vkbPhysicalDevice.physical_device);

    vkb::DeviceBuilder deviceBuilder{phyRet.value()};
    const auto deviceRet = deviceBuilder.build();
    if (!deviceRet)
    {
        throw std::runtime_error("failed to create device");
    }

    // device
    vkb::Device vkbDevice = deviceRet.value();
    device = vk::raii::Device(physicalDevice, vkbDevice.device);

    // cleanup vkbooststrap
    vkbInstance.instance = VK_NULL_HANDLE;
    vkbInstance.debug_messenger = VK_NULL_HANDLE;
    vkbPhysicalDevice.physical_device = VK_NULL_HANDLE;
    vkbDevice.device = VK_NULL_HANDLE;
}

VkContext::~VkContext()
{
}

} // namespace vk_engine