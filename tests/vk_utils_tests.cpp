#include "vk_utils.h"

#include <cassert>
#include <stdexcept>
#include <string>

int main()
{
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
