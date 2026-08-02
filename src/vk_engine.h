#pragma once

#include "vk_context.h"
#include "vk_buffer.h"
#include "vk_pipeline.h"
#include "vk_renderer.h"
#include "vk_swapchain.h"
#include "vk_vertex.h"
#include "vk_window.h"
#include <memory>

namespace vk_engine
{
class VkEngine
{
public:
    VkEngine();
    ~VkEngine();

    void Run();

private:
    std::unique_ptr<VkWindow> window{nullptr};
    std::unique_ptr<VkContext> context{nullptr};
    std::unique_ptr<VkSwapchain> swapchain{nullptr};
    std::unique_ptr<VkBuffer> triangleVertexBuffer{nullptr};
    std::unique_ptr<GraphicsPipeline> graphicsPipeline{nullptr};
    std::unique_ptr<VkRenderer> renderer{nullptr};
};
} // namespace vk_engine
