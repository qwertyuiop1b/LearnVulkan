#include "vk_utils.h"
#include "vk_buffer.h"
#include "vk_engine.h"
#include "vk_pipeline.h"
#include "vk_renderer.h"
#include "vk_shader.h"
#include "vk_swapchain.h"

#include <cassert>
#include <concepts>
#include <functional>
#include <fstream>
#include <filesystem>
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

    static_assert(!std::is_copy_constructible_v<vk_engine::VkFrameContext>);
    static_assert(!std::is_copy_assignable_v<vk_engine::VkFrameContext>);
    static_assert(!std::is_copy_constructible_v<vk_engine::VkRenderer>);
    static_assert(!std::is_copy_assignable_v<vk_engine::VkRenderer>);
    static_assert(!std::is_move_constructible_v<vk_engine::VkRenderer>);
    static_assert(!std::is_move_assignable_v<vk_engine::VkRenderer>);

    using ExpectedRenderCallback = std::function<void(vk::CommandBuffer, vk_engine::RenderHelper&)>;
    static_assert(std::same_as<vk_engine::VkRenderer::RenderCallback, ExpectedRenderCallback>);
    using ExpectedEngineRun = void (vk_engine::VkEngine::*)(const vk_engine::VkRenderer::RenderCallback&);
    static_assert(std::same_as<decltype(&vk_engine::VkEngine::Run), ExpectedEngineRun>);

    static_assert(!std::is_copy_constructible_v<vk_engine::Buffer>);
    static_assert(!std::is_copy_assignable_v<vk_engine::Buffer>);
    static_assert(!std::is_copy_constructible_v<vk_engine::ShaderModule>);
    static_assert(!std::is_copy_assignable_v<vk_engine::ShaderModule>);
    static_assert(!std::is_copy_constructible_v<vk_engine::GraphicsPipeline>);
    static_assert(!std::is_copy_assignable_v<vk_engine::GraphicsPipeline>);

    const vk_engine::GraphicsPipelineDescription pipelineDescription{};
    assert(pipelineDescription.vertexShader.empty());
    assert(pipelineDescription.fragmentShader.empty());
    assert(pipelineDescription.topology == vk::PrimitiveTopology::eTriangleList);
    assert(pipelineDescription.polygonMode == vk::PolygonMode::eFill);
    assert(pipelineDescription.cullMode == vk::CullModeFlags{});
    assert(pipelineDescription.samples == vk::SampleCountFlagBits::e1);
    assert(!pipelineDescription.depthTestEnable);
    assert(!pipelineDescription.depthWriteEnable);
    assert(!pipelineDescription.blendEnable);

    assert(vk_engine::ShaderPath("test.spv") == std::filesystem::path{"shaders"} / "test.spv");

    const std::filesystem::path shaderTestDirectory =
        std::filesystem::temp_directory_path() / "learnvulkan_shader_tests";
    std::filesystem::create_directories(shaderTestDirectory);
    const auto emptyShaderPath = shaderTestDirectory / "empty.spv";
    const auto malformedShaderPath = shaderTestDirectory / "malformed.spv";
    const auto invalidMagicShaderPath = shaderTestDirectory / "invalid-magic.spv";
    const auto validShaderPath = shaderTestDirectory / "valid.spv";
    std::ofstream(emptyShaderPath, std::ios::binary).close();
    std::ofstream malformedShader(malformedShaderPath, std::ios::binary);
    malformedShader.put('\x01');
    malformedShader.close();
    const uint32_t invalidMagic = 0x01020304U;
    std::ofstream invalidMagicShader(invalidMagicShaderPath, std::ios::binary);
    invalidMagicShader.write(reinterpret_cast<const char*>(&invalidMagic), sizeof(invalidMagic));
    invalidMagicShader.close();
    const std::array<uint32_t, 5> validCode{0x07230203U, 0x00010000U, 0U, 0U, 0U};
    std::ofstream validShader(validShaderPath, std::ios::binary);
    validShader.write(reinterpret_cast<const char*>(validCode.data()),
                      static_cast<std::streamsize>(validCode.size() * sizeof(uint32_t)));
    validShader.close();

    const auto expectShaderError = [](const std::filesystem::path& path)
    {
        try
        {
            (void)vk_engine::ReadSpirvFile(path);
            assert(false && "invalid SPIR-V input must throw");
        }
        catch (const std::runtime_error& error)
        {
            assert(std::string{error.what()}.find(path.string()) != std::string::npos);
        }
    };
    expectShaderError(emptyShaderPath);
    expectShaderError(malformedShaderPath);
    expectShaderError(invalidMagicShaderPath);
    const std::vector<uint32_t> expectedCode(validCode.begin(), validCode.end());
    assert(vk_engine::ReadSpirvFile(validShaderPath) == expectedCode);
    std::filesystem::remove_all(shaderTestDirectory);

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
