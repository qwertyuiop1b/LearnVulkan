#pragma once


#include "Qdevice.hpp"
#include "Qpipeline.hpp"
#include "Qwindow.hpp"
#include "Qswapchain.hpp"
#include "vulkan/vulkan_core.h"

#include <memory>
#include <vector>

namespace q_vulkan {

class FirstApp {
public:

    FirstApp();
    ~FirstApp();

    FirstApp(const FirstApp&) = delete;
    FirstApp& operator=(const FirstApp&) = delete;


    void run() {
        while(!window.shouldClose()) {
            
            window.pollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device.device());
    }

    

private:
    void createPipelineLayout();
    void createPipeline() ;
    void createCommandBuffer();
    void drawFrame();

    static constexpr int width = 800;
    static constexpr int height = 600;
    QWindow window {width, height,  "first app"};
    QDevice device {window};
    QSwapChain swapchain { device, window.getExtent()};
    std::unique_ptr<QPipeline> pipeline;
    VkPipelineLayout pipelineLayout;
    std::vector<VkCommandBuffer> commmandBuffers;
};

};