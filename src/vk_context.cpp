#include "vk_context.h"
#include "VkBootstrap.h"
#include <iostream>
#include <stdexcept>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>


namespace vk_engine {
VkContext::VkContext() : window(window)
{
    vkb::InstanceBuilder builder;
    auto vkbInstanceRet = builder
        .set_app_name("vk-engine")
        .set_app_version(1, 0, 0)
        .set_engine_name("No Engine")
        .set_engine_version(1, 0, 0)
        .require_api_version(1, 3, 0)
        .request_validation_layers()
        .build();
    if (!vkbInstanceRet) {
        throw std::runtime_error("failed to create instance");
    }

    vkb::Instance vkbInstance = vkbInstanceRet.value();    
    vk::raii::Context context;
    instance = vk::raii::Instance(context, vkbInstance.instance);

    // surface
    VkSurfaceKHR rawSurface;
    glfwCreateWindowSurface(vkbInstance.instance, window, nullptr, &rawSurface);
    surface = vk::raii::SurfaceKHR(instance, rawSurface);

    vkb::PhysicalDeviceSelector selector {vkbInstance};
    auto phyRet = selector.set_surface(*surface).select();
    if (!phyRet) {
        throw std::runtime_error("failed to select physical");
    }

    vkb::PhysicalDevice vkbPhysicalDevice = phyRet.value();
    physicalDevice = vk::raii::PhysicalDevice(instance, vkbPhysicalDevice.physical_device);

    vkb::DeviceBuilder deviceBuilder {phyRet.value()};
    auto deviceRet = deviceBuilder.build();
    if (!deviceRet) {
        throw std::runtime_error("failed to create device");
    }

    vkb::Device vkbDevice = deviceRet.value();
    device = vk::raii::Device(physicalDevice, vkbDevice.device);


}

VkContext::~VkContext() {

}

}