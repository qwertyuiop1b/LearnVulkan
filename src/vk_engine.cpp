#include "vk_engine.h"
#include "vk_context.h"
#include "vk_window.h"
#include <memory>

namespace vk_engine
{
VkEngine::VkEngine()
{
    window = std::make_unique<VkWindow>(800, 600);
    context = std::make_unique<VkContext>(*window.get());
}

VkEngine::~VkEngine()
{
}

void VkEngine::Run()
{
    while (!window->ShouldClose())
    {

        window->ProcessPendingEvents();
    }
}
} // namespace vk_engine
