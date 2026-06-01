#pragma once

/**
 * @file texture_loader.hpp
 * @brief 图像加载（stb_image）与 VkImage 上传工具
 */

#include <vulkan_tutorial/asset_path.hpp>
#include <vulkan_tutorial/utils.hpp>

#include "../../external/stb/stb_image.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vulkan_tutorial {

struct ImageData {
    std::vector<uint8_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
};

struct TextureImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 1;
};

inline ImageData loadImageFromFile(const std::string& path, int desiredChannels = 0) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("无法打开图像: " + path);
    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint8_t> buffer(fileSize);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fileSize));
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(
        buffer.data(), static_cast<int>(buffer.size()), &width, &height, &channels, desiredChannels);
    if (!pixels)
        throw std::runtime_error("stb_image 解析失败: " + path);
    const int outChannels = desiredChannels != 0 ? desiredChannels : channels;
    ImageData result;
    result.width = static_cast<uint32_t>(width);
    result.height = static_cast<uint32_t>(height);
    result.channels = static_cast<uint32_t>(outChannels);
    result.pixels.resize(static_cast<size_t>(width * height * outChannels));
    std::memcpy(result.pixels.data(), pixels, static_cast<size_t>(width * height * outChannels));
    stbi_image_free(pixels);
    return result;
}

inline ImageData generateCheckerboard(uint32_t size, uint32_t tileSize) {
    ImageData img;
    img.width = size;
    img.height = size;
    img.channels = 4;
    img.pixels.resize(static_cast<size_t>(size * size * 4));
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            const bool dark = ((x / tileSize) + (y / tileSize)) % 2 == 0;
            const size_t i = static_cast<size_t>((y * size + x) * 4);
            img.pixels[i + 0] = dark ? 40 : 220;
            img.pixels[i + 1] = dark ? 40 : 220;
            img.pixels[i + 2] = dark ? 40 : 220;
            img.pixels[i + 3] = 255;
        }
    }
    return img;
}

inline ImageData generateBrickDiffuse(uint32_t size) {
    ImageData img;
    img.width = size;
    img.height = size;
    img.channels = 4;
    img.pixels.resize(static_cast<size_t>(size * size * 4));
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            const uint32_t row = y / (size / 8);
            const uint32_t col = (x + (row % 2) * (size / 16)) / (size / 4);
            const bool mortar = (y % (size / 8) < 2) || (x % (size / 4) < 2);
            const size_t i = static_cast<size_t>((y * size + x) * 4);
            if (mortar) {
                img.pixels[i + 0] = 180;
                img.pixels[i + 1] = 175;
                img.pixels[i + 2] = 165;
            } else {
                img.pixels[i + 0] = static_cast<uint8_t>(140 + (col * 17 % 40));
                img.pixels[i + 1] = static_cast<uint8_t>(60 + (row * 13 % 30));
                img.pixels[i + 2] = static_cast<uint8_t>(40 + (col * 7 % 20));
            }
            img.pixels[i + 3] = 255;
        }
    }
    return img;
}

inline ImageData generateBrickNormal(uint32_t size) {
    ImageData img;
    img.width = size;
    img.height = size;
    img.channels = 4;
    img.pixels.resize(static_cast<size_t>(size * size * 4));
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(size);
            const float v = static_cast<float>(y) / static_cast<float>(size);
            const float bump = std::sin(u * 40.0f) * std::cos(v * 40.0f) * 0.15f;
            const float nx = bump;
            const float ny = bump * 0.5f;
            const float nz = std::sqrt(std::max(0.0f, 1.0f - nx * nx - ny * ny));
            const size_t i = static_cast<size_t>((y * size + x) * 4);
            img.pixels[i + 0] = static_cast<uint8_t>((nx * 0.5f + 0.5f) * 255.0f);
            img.pixels[i + 1] = static_cast<uint8_t>((ny * 0.5f + 0.5f) * 255.0f);
            img.pixels[i + 2] = static_cast<uint8_t>((nz * 0.5f + 0.5f) * 255.0f);
            img.pixels[i + 3] = 255;
        }
    }
    return img;
}

inline ImageData generateSkyGradient(uint32_t size) {
    ImageData img;
    img.width = size;
    img.height = size;
    img.channels = 4;
    img.pixels.resize(static_cast<size_t>(size * size * 4));
    for (uint32_t y = 0; y < size; ++y) {
        const float t = static_cast<float>(y) / static_cast<float>(size);
        for (uint32_t x = 0; x < size; ++x) {
            const size_t i = static_cast<size_t>((y * size + x) * 4);
            img.pixels[i + 0] = static_cast<uint8_t>((1.0f - t) * 80 + t * 20);
            img.pixels[i + 1] = static_cast<uint8_t>((1.0f - t) * 140 + t * 60);
            img.pixels[i + 2] = static_cast<uint8_t>((1.0f - t) * 220 + t * 120);
            img.pixels[i + 3] = 255;
        }
    }
    return img;
}

inline VkFormat channelsToFormat(uint32_t channels) {
    if (channels == 4)
        return VK_FORMAT_R8G8B8A8_SRGB;
    if (channels == 3)
        return VK_FORMAT_R8G8B8_SRGB;
    if (channels == 1)
        return VK_FORMAT_R8_UNORM;
    throw std::runtime_error("不支持的通道数");
}

inline uint32_t
findMemoryTypeIndex(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    throw std::runtime_error("找不到合适的内存类型");
}

inline void transitionImageLayout(VkDevice device,
                                  VkCommandPool commandPool,
                                  VkQueue queue,
                                  VkImage image,
                                  VkFormat format,
                                  VkImageLayout oldLayout,
                                  VkImageLayout newLayout,
                                  uint32_t mipLevels = 1) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        (void)format;
        throw std::runtime_error("不支持的布局转换");
    }
    vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

inline void copyBufferToImage(VkDevice device,
                              VkCommandPool commandPool,
                              VkQueue queue,
                              VkBuffer buffer,
                              VkImage image,
                              uint32_t width,
                              uint32_t height) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

inline TextureImage createTextureFromImageData(VkPhysicalDevice physicalDevice,
                                               VkDevice device,
                                               VkCommandPool commandPool,
                                               VkQueue queue,
                                               const ImageData& imageData,
                                               bool srgb = true) {
    TextureImage texture{};
    texture.width = imageData.width;
    texture.height = imageData.height;
    texture.mipLevels = 1;
    texture.format = srgb ? channelsToFormat(imageData.channels) : VK_FORMAT_R8G8B8A8_UNORM;
    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(imageData.width * imageData.height * imageData.channels);
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer));
    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memReq);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex =
        findMemoryTypeIndex(physicalDevice,
                            memReq.memoryTypeBits,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &stagingMemory));
    VK_CHECK(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0));
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(device, stagingMemory, 0, imageSize, 0, &mapped));
    std::memcpy(mapped, imageData.pixels.data(), static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingMemory);
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {imageData.width, imageData.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = texture.format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &texture.image));
    vkGetImageMemoryRequirements(device, texture.image, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex =
        findMemoryTypeIndex(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &texture.memory));
    VK_CHECK(vkBindImageMemory(device, texture.image, texture.memory, 0));
    transitionImageLayout(device,
                          commandPool,
                          queue,
                          texture.image,
                          texture.format,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(device, commandPool, queue, stagingBuffer, texture.image, imageData.width, imageData.height);
    transitionImageLayout(device,
                          commandPool,
                          queue,
                          texture.image,
                          texture.format,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = texture.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = texture.format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &texture.view));
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &texture.sampler));
    return texture;
}

inline void destroyTexture(VkDevice device, TextureImage& texture) {
    if (texture.sampler != VK_NULL_HANDLE)
        vkDestroySampler(device, texture.sampler, nullptr);
    if (texture.view != VK_NULL_HANDLE)
        vkDestroyImageView(device, texture.view, nullptr);
    if (texture.image != VK_NULL_HANDLE)
        vkDestroyImage(device, texture.image, nullptr);
    if (texture.memory != VK_NULL_HANDLE)
        vkFreeMemory(device, texture.memory, nullptr);
    texture = {};
}

inline ImageData loadImageWithFallback(const std::string& relativePath, const ImageData& fallback) {
    const std::string path = resolveAssetPath(relativePath);
    try {
        return loadImageFromFile(path);
    } catch (const std::exception&) {
        return fallback;
    }
}

} // namespace vulkan_tutorial
