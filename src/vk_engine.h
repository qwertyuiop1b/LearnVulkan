#pragma once

#include "vk_context.h"
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
};
} // namespace vk_engine
