#include <graphics/memory/texture.hpp>

#include <stdexcept>

namespace vulkan_graphics {

Texture::Texture(VulkanAllocator& allocator, UploadContext& uploads, const TextureCreateInfo& createInfo) {
    if (createInfo.extent.width == 0 || createInfo.extent.height == 0)
        throw std::invalid_argument("Texture extent must be greater than zero");
    if (createInfo.pixelData == nullptr || createInfo.dataSize == 0)
        throw std::invalid_argument("Texture requires pixel data");
    if (createInfo.format == vk::Format::eUndefined)
        throw std::invalid_argument("Texture format must be specified");

    ImageCreateInfo imageInfo{};
    imageInfo.extent = createInfo.extent;
    imageInfo.format = createInfo.format;
    imageInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
    imageInfo.aspectMask = vk::ImageAspectFlagBits::eColor;
    image_ = Image{allocator, imageInfo};

    BufferCreateInfo stagingInfo{};
    stagingInfo.size = createInfo.dataSize;
    stagingInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
    stagingInfo.memoryUsage = BufferMemoryUsage::CpuToGpu;
    Buffer stagingBuffer{allocator, stagingInfo};
    stagingBuffer.write(createInfo.pixelData, createInfo.dataSize);

    uploads.transitionImageLayout(image_, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    uploads.copyBufferToImage(stagingBuffer, image_);
    uploads.transitionImageLayout(image_, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

    sampler_ = Sampler{allocator.context(), createInfo.sampler};
}

bool Texture::isValid() const noexcept {
    return image_.isValid() && sampler_.isValid();
}

const Image& Texture::image() const noexcept {
    return image_;
}

const Sampler& Texture::sampler() const noexcept {
    return sampler_;
}

} // namespace vulkan_graphics
