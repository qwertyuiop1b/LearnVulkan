#pragma once

#include <graphics/memory/buffer.hpp>
#include <graphics/memory/texture.hpp>
#include <graphics/shader/descriptor_set_layout.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace vulkan_graphics {

class DescriptorPool;
struct DescriptorPoolState;

class DescriptorSet final {
  public:
    DescriptorSet() = default;
    ~DescriptorSet() noexcept;

    DescriptorSet(const DescriptorSet&) = delete;
    DescriptorSet& operator=(const DescriptorSet&) = delete;
    DescriptorSet(DescriptorSet&& other) noexcept;
    DescriptorSet& operator=(DescriptorSet&& other) noexcept;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] vk::DescriptorSet handle() const noexcept;
    [[nodiscard]] VkDescriptorSet nativeHandle() const noexcept;

    void writeBuffer(uint32_t binding,
                     const Buffer& buffer,
                     VkDeviceSize offset = 0,
                     VkDeviceSize range = VK_WHOLE_SIZE,
                     uint32_t arrayElement = 0);
    void writeTexture(uint32_t binding, const Texture& texture, uint32_t arrayElement = 0);

  private:
    friend class DescriptorPool;

    DescriptorSet(std::shared_ptr<DescriptorPoolState> poolState,
                  VkDescriptorSet descriptorSet,
                  std::vector<DescriptorBinding> bindings);

    [[nodiscard]] const DescriptorBinding& findBinding(uint32_t binding, uint32_t arrayElement) const;
    void release() noexcept;

    std::shared_ptr<DescriptorPoolState> poolState_;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    std::vector<DescriptorBinding> bindings_;
};

} // namespace vulkan_graphics
