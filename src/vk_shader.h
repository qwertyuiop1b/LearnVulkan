#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

namespace vk_engine
{
inline std::filesystem::path ShaderPath(std::string_view filename)
{
#ifdef VK_ENGINE_SHADER_DIR
    return std::filesystem::path{VK_ENGINE_SHADER_DIR} / std::string{filename};
#else
    return std::filesystem::path{"shaders"} / std::string{filename};
#endif
}

std::vector<uint32_t> ReadSpirvFile(const std::filesystem::path& path);

class ShaderModule
{
public:
    ShaderModule(const vk::raii::Device& device, const std::filesystem::path& path);
    ~ShaderModule() = default;

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule(ShaderModule&&) = delete;
    ShaderModule& operator=(ShaderModule&&) = delete;

    vk::ShaderModule GetHandle() const noexcept
    {
        return *module;
    }

private:
    vk::raii::ShaderModule module{nullptr};
};
} // namespace vk_engine
