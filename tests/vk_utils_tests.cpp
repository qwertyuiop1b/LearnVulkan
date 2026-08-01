#include "vk_utils.h"
#include "vk_renderer.h"
#include "vk_swapchain.h"

#include <cassert>
#include <concepts>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

int main()
{
    static_assert(!std::is_copy_constructible_v<vk_engine::VkSwapchain>);
    static_assert(std::same_as<decltype(std::declval<const vk_engine::VkSwapchain&>().GetImageCount()), uint32_t>);
    static_assert(std::same_as<decltype(std::declval<const vk_engine::VkSwapchain&>().GetImages()),
                               const std::vector<vk::Image>&>);
    static_assert(std::same_as<decltype(std::declval<const vk_engine::VkSwapchain&>().GetImage(0)), vk::Image>);
    static_assert(std::same_as<decltype(std::declval<const vk_engine::VkSwapchain&>().GetImageView(0)), vk::ImageView>);

    static_assert(!std::is_copy_constructible_v<vk_engine::VkFrameContext>);
    static_assert(!std::is_copy_assignable_v<vk_engine::VkFrameContext>);
    static_assert(!std::is_copy_constructible_v<vk_engine::VkRenderer>);
    static_assert(!std::is_copy_assignable_v<vk_engine::VkRenderer>);
    static_assert(!std::is_move_constructible_v<vk_engine::VkRenderer>);
    static_assert(!std::is_move_assignable_v<vk_engine::VkRenderer>);

    using ExpectedRecordCallback = std::function<void(vk::CommandBuffer, vk::Image, vk::ImageView, vk::Extent2D)>;
    static_assert(std::same_as<vk_engine::VkRenderer::RecordCallback, ExpectedRecordCallback>);

    int invocationCount = 0;
    VK_CHECK((
        [&]() -> VkResult
        {
            ++invocationCount;
            return VK_SUCCESS;
        })());
    assert(invocationCount == 1);

    try
    {
        VK_CHECK(VK_ERROR_DEVICE_LOST);
        assert(false && "VK_CHECK must throw for a failed VkResult");
    }
    catch (const std::runtime_error& error)
    {
        const std::string message = error.what();
        assert(message.find("VK_ERROR_DEVICE_LOST") != std::string::npos);
        assert(message.find("-4") != std::string::npos);
    }
}
