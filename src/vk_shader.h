#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

namespace vk_engine
{
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
