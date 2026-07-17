#include <graphics/shader/shader_module.hpp>

#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vulkan_graphics {

ShaderModule::ShaderModule(const VulkanContext& context, const uint32_t* spirvWords, size_t wordCount) : context_(&context) {
    if (static_cast<VkDevice>(context.device()) == VK_NULL_HANDLE)
        throw std::invalid_argument("ShaderModule requires an initialized VulkanContext");
    if (spirvWords == nullptr || wordCount == 0)
        throw std::invalid_argument("ShaderModule requires non-empty SPIR-V code");
    if (wordCount > std::numeric_limits<size_t>::max() / sizeof(uint32_t))
        throw std::invalid_argument("SPIR-V code is too large");

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = wordCount * sizeof(uint32_t);
    createInfo.pCode = spirvWords;
    if (vkCreateShaderModule(static_cast<VkDevice>(context.device()), &createInfo, nullptr, &shaderModule_) != VK_SUCCESS) {
        context_ = nullptr;
        throw std::runtime_error("Failed to create Vulkan shader module");
    }
}

ShaderModule::ShaderModule(const VulkanContext& context, const std::vector<uint32_t>& spirvWords)
    : ShaderModule(context, spirvWords.data(), spirvWords.size()) {}

ShaderModule::~ShaderModule() noexcept {
    destroy();
}

ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      shaderModule_(std::exchange(other.shaderModule_, VK_NULL_HANDLE)) {}

ShaderModule& ShaderModule::operator=(ShaderModule&& other) noexcept {
    if (this == &other)
        return *this;

    destroy();
    context_ = std::exchange(other.context_, nullptr);
    shaderModule_ = std::exchange(other.shaderModule_, VK_NULL_HANDLE);
    return *this;
}

ShaderModule ShaderModule::fromFile(const VulkanContext& context, const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open SPIR-V file: " + path.string());

    const std::streamsize fileSize = file.tellg();
    if (fileSize <= 0 || fileSize % static_cast<std::streamsize>(sizeof(uint32_t)) != 0)
        throw std::runtime_error("SPIR-V file size must be a positive multiple of four bytes: " + path.string());

    std::vector<uint32_t> spirvWords(static_cast<size_t>(fileSize) / sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(spirvWords.data()), fileSize))
        throw std::runtime_error("Failed to read SPIR-V file: " + path.string());
    if (spirvWords.front() != 0x07230203)
        throw std::runtime_error("SPIR-V file has an invalid magic number: " + path.string());

    return ShaderModule{context, spirvWords};
}

bool ShaderModule::isValid() const noexcept {
    return shaderModule_ != VK_NULL_HANDLE;
}

vk::ShaderModule ShaderModule::handle() const noexcept {
    return vk::ShaderModule{shaderModule_};
}

VkShaderModule ShaderModule::nativeHandle() const noexcept {
    return shaderModule_;
}

void ShaderModule::destroy() noexcept {
    if (shaderModule_ != VK_NULL_HANDLE)
        vkDestroyShaderModule(static_cast<VkDevice>(context_->device()), shaderModule_, nullptr);

    context_ = nullptr;
    shaderModule_ = VK_NULL_HANDLE;
}

} // namespace vulkan_graphics
