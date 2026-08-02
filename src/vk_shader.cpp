#include "vk_shader.h"

#include <fstream>
#include <stdexcept>

namespace vk_engine
{
namespace
{
constexpr uint32_t kSpirvMagic = 0x07230203U;

[[noreturn]] void ThrowShaderFileError(const std::filesystem::path& path, const char* reason)
{
    throw std::runtime_error("failed to load SPIR-V '" + path.string() + "': " + reason);
}
} // namespace

std::vector<uint32_t> ReadSpirvFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        ThrowShaderFileError(path, "file could not be opened");
    }

    const std::streampos end = file.tellg();
    if (end <= std::streampos{0})
    {
        ThrowShaderFileError(path, "file is empty");
    }

    const std::size_t byteCount = static_cast<std::size_t>(end);
    if (byteCount % sizeof(uint32_t) != 0)
    {
        ThrowShaderFileError(path, "file size is not a multiple of 4 bytes");
    }

    std::vector<uint32_t> code(byteCount / sizeof(uint32_t));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(byteCount));
    if (!file)
    {
        ThrowShaderFileError(path, "file could not be read");
    }

    if (code.front() != kSpirvMagic)
    {
        ThrowShaderFileError(path, "SPIR-V magic is invalid");
    }

    return code;
}

ShaderModule::ShaderModule(const vk::raii::Device& device, const std::filesystem::path& path)
{
    const std::vector<uint32_t> code = ReadSpirvFile(path);

    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.setCode(code);
    module = vk::raii::ShaderModule(device, createInfo);
}
} // namespace vk_engine
