#include <graphics/command/frame_context.hpp>
#include <graphics/core/vulkan_context.hpp>
#include <graphics/memory/vulkan_allocator.hpp>
#include <graphics/pipeline/graphics_pipeline.hpp>
#include <graphics/render/frame_scheduler.hpp>
#include <graphics/render/swapchain.hpp>
#include <graphics/shader/pipeline_layout.hpp>
#include <graphics/shader/shader_module.hpp>

#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace vulkan_graphics;

class VkitTriangleApp {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window_ = nullptr;
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;

    std::unique_ptr<VulkanContext> context_;
    std::unique_ptr<VulkanAllocator> allocator_;
    std::unique_ptr<Swapchain> swapchain_;
    std::unique_ptr<FrameScheduler> frameScheduler_;
    std::unique_ptr<ShaderModule> vertexShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<PipelineLayout> pipelineLayout_;
    std::unique_ptr<GraphicsPipeline> pipeline_;

    void initWindow() {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window_ = glfwCreateWindow(WIDTH, HEIGHT, "VkKit Triangle", nullptr, nullptr);
        if (!window_) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/) {
        // 窗口大小改变时的回调，实际重建会在渲染循环中处理
    }

    void initVulkan() {
        // 创建 Vulkan Context
        VulkanContextCreateInfo contextInfo{};
        contextInfo.window = window_;
        contextInfo.applicationName = "VkKit Triangle";
        contextInfo.enableValidation = true;
        contextInfo.requireDynamicRendering = true;
        contextInfo.requireSynchronization2 = true;
        context_ = std::make_unique<VulkanContext>(contextInfo);

        // 创建内存分配器
        allocator_ = std::make_unique<VulkanAllocator>(*context_);

        // 创建交换链
        SwapchainCreateInfo swapchainInfo{};
        swapchainInfo.desiredExtent = {WIDTH, HEIGHT};
        swapchain_ = std::make_unique<Swapchain>(*context_, swapchainInfo);

        // 创建帧调度器
        FrameSchedulerCreateInfo schedulerInfo{};
        schedulerInfo.swapchain = swapchain_.get();
        schedulerInfo.framesInFlight = 2;
        frameScheduler_ = std::make_unique<FrameScheduler>(*context_, schedulerInfo);

        // 加载着色器
        auto vertShader = ShaderModule::fromFile(*context_, "shaders/vkit_triangle.vert.spv");
        auto fragShader = ShaderModule::fromFile(*context_, "shaders/vkit_triangle.frag.spv");
        vertexShader_ = std::make_unique<ShaderModule>(std::move(vertShader));
        fragmentShader_ = std::make_unique<ShaderModule>(std::move(fragShader));

        // 创建管线布局（无描述符集，无推送常量）
        PipelineLayoutCreateInfo layoutInfo{};
        pipelineLayout_ = std::make_unique<PipelineLayout>(*context_, layoutInfo);

        // 创建图形管线
        GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.layout = pipelineLayout_.get();
        pipelineInfo.shaderStages = {
            {vertexShader_.get(), vk::ShaderStageFlagBits::eVertex, "main"},
            {fragmentShader_.get(), vk::ShaderStageFlagBits::eFragment, "main"},
        };

        // 无顶点输入（顶点数据硬编码在着色器中）
        pipelineInfo.vertexBindings = {};
        pipelineInfo.vertexAttributes = {};

        // 设置颜色附件格式
        pipelineInfo.colorAttachmentFormats = {swapchain_->format()};

        // 其他默认状态
        pipelineInfo.topology = vk::PrimitiveTopology::eTriangleList;
        pipelineInfo.cullMode = vk::CullModeFlagBits::eNone;
        pipelineInfo.frontFace = vk::FrontFace::eCounterClockwise;
        pipelineInfo.depthTestEnable = false;
        pipelineInfo.depthWriteEnable = false;

        pipeline_ = std::make_unique<GraphicsPipeline>(*context_, pipelineInfo);
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            drawFrame();
        }

        // 等待设备空闲后再清理
        context_->waitIdle();
    }

    void drawFrame() {
        // 开始帧
        const FrameBeginResult beginResult = frameScheduler_->beginFrame();

        if (beginResult.status == SwapchainStatus::eOutOfDate) {
            // 交换链过期，重建
            int width, height;
            glfwGetFramebufferSize(window_, &width, &height);
            while (width == 0 || height == 0) {
                glfwGetFramebufferSize(window_, &width, &height);
                glfwWaitEvents();
            }
            frameScheduler_->recreateSwapchain({static_cast<uint32_t>(width),
                                                static_cast<uint32_t>(height)});
            return;
        }

        VkCommandBuffer commandBuffer = beginResult.frame->commandBuffer();

        // 开始动态渲染
        DynamicRenderingInfo renderingInfo{};
        renderingInfo.colorClearValue = {{0.0f, 0.0f, 0.0f, 1.0f}};
        frameScheduler_->beginDynamicRendering(renderingInfo);

        // 设置视口和裁剪矩形
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(beginResult.renderTarget->extent().width);
        viewport.height = static_cast<float>(beginResult.renderTarget->extent().height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = beginResult.renderTarget->extent();
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // 绑定管线
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                         pipeline_->nativeHandle());

        // 绘制三角形（3个顶点，无顶点缓冲区）
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        // 结束动态渲染
        frameScheduler_->endDynamicRendering();

        // 结束帧并呈现
        const SwapchainStatus presentStatus = frameScheduler_->endFrame();

        if (presentStatus == SwapchainStatus::eOutOfDate ||
            presentStatus == SwapchainStatus::eSuboptimal) {
            int width, height;
            glfwGetFramebufferSize(window_, &width, &height);
            if (width > 0 && height > 0) {
                frameScheduler_->recreateSwapchain({static_cast<uint32_t>(width),
                                                    static_cast<uint32_t>(height)});
            }
        }
    }

    void cleanup() {
        if (context_) {
            context_->waitIdle();
        }

        // 智能指针会自动按正确的顺序销毁
        // 顺序：pipeline_ -> pipelineLayout_ -> shaders -> frameScheduler_
        //      -> swapchain_ -> allocator_ -> context_
        pipeline_.reset();
        pipelineLayout_.reset();
        fragmentShader_.reset();
        vertexShader_.reset();
        frameScheduler_.reset();
        swapchain_.reset();
        allocator_.reset();
        context_.reset();

        if (window_) {
            glfwDestroyWindow(window_);
        }
        glfwTerminate();
    }
};

int main() {
    VkitTriangleApp app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
