#include <graphics/buffer.hpp>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace vulkan_graphics {
namespace {

VmaMemoryUsage toVmaMemoryUsage(BufferMemoryUsage usage) {
    switch (usage) {
    case BufferMemoryUsage::GpuOnly:
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    case BufferMemoryUsage::CpuToGpu:
    case BufferMemoryUsage::GpuToCpu:
        return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    }

    throw std::invalid_argument("Unknown buffer memory usage");
}

VmaAllocationCreateFlags toVmaHostAccessFlags(BufferMemoryUsage memoryUsage, BufferHostAccess hostAccess) {
    if (memoryUsage == BufferMemoryUsage::GpuOnly) {
        if (hostAccess != BufferHostAccess::None)
            throw std::invalid_argument("GPU-only buffers cannot request host access");
        return 0;
    }

    if (hostAccess == BufferHostAccess::None)
        hostAccess = memoryUsage == BufferMemoryUsage::CpuToGpu ? BufferHostAccess::SequentialWrite
                                                                 : BufferHostAccess::Random;

    switch (hostAccess) {
    case BufferHostAccess::SequentialWrite:
        return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    case BufferHostAccess::Random:
        return VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    case BufferHostAccess::None:
        break;
    }

    throw std::invalid_argument("Host access must be specified for host-visible buffers");
}

} // namespace

Buffer::Buffer(VulkanAllocator& allocator, const BufferCreateInfo& createInfo) : allocator_(&allocator) {
    if (!allocator.isValid())
        throw std::invalid_argument("Buffer requires a valid VulkanAllocator");
    if (createInfo.size == 0)
        throw std::invalid_argument("Buffer size must be greater than zero");
    if (createInfo.usage == vk::BufferUsageFlags{})
        throw std::invalid_argument("Buffer usage flags must not be empty");
    if (createInfo.persistentMap && createInfo.memoryUsage == BufferMemoryUsage::GpuOnly)
        throw std::invalid_argument("GPU-only buffers cannot be persistently mapped");

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = createInfo.size;
    bufferInfo.usage = static_cast<VkBufferUsageFlags>(createInfo.usage);
    bufferInfo.sharingMode = static_cast<VkSharingMode>(createInfo.sharingMode);
    bufferInfo.queueFamilyIndexCount = static_cast<uint32_t>(createInfo.queueFamilyIndices.size());
    bufferInfo.pQueueFamilyIndices = createInfo.queueFamilyIndices.data();

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = toVmaMemoryUsage(createInfo.memoryUsage);
    allocationInfo.flags = toVmaHostAccessFlags(createInfo.memoryUsage, createInfo.hostAccess);
    if (createInfo.persistentMap)
        allocationInfo.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo createdAllocationInfo{};
    const VkResult result = vmaCreateBuffer(
        allocator.nativeHandle(), &bufferInfo, &allocationInfo, &buffer_, &allocation_, &createdAllocationInfo);
    if (result != VK_SUCCESS) {
        allocator_ = nullptr;
        throw std::runtime_error("Failed to create Vulkan buffer");
    }

    size_ = createInfo.size;
    usage_ = createInfo.usage;
    hostAccessible_ = createInfo.memoryUsage != BufferMemoryUsage::GpuOnly;
    persistentMappedData_ = createdAllocationInfo.pMappedData;
}

Buffer::~Buffer() noexcept {
    destroy();
}

Buffer::Buffer(Buffer&& other) noexcept
    : allocator_(std::exchange(other.allocator_, nullptr)),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      size_(std::exchange(other.size_, 0)),
      usage_(std::exchange(other.usage_, {})),
      hostAccessible_(std::exchange(other.hostAccessible_, false)),
      persistentMappedData_(std::exchange(other.persistentMappedData_, nullptr)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this == &other)
        return *this;

    destroy();
    allocator_ = std::exchange(other.allocator_, nullptr);
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    size_ = std::exchange(other.size_, 0);
    usage_ = std::exchange(other.usage_, {});
    hostAccessible_ = std::exchange(other.hostAccessible_, false);
    persistentMappedData_ = std::exchange(other.persistentMappedData_, nullptr);
    return *this;
}

bool Buffer::isValid() const noexcept {
    return buffer_ != VK_NULL_HANDLE;
}

bool Buffer::isPersistentlyMapped() const noexcept {
    return persistentMappedData_ != nullptr;
}

vk::Buffer Buffer::handle() const noexcept {
    return vk::Buffer{buffer_};
}

VkBuffer Buffer::nativeHandle() const noexcept {
    return buffer_;
}

VkDeviceSize Buffer::size() const noexcept {
    return size_;
}

vk::BufferUsageFlags Buffer::usage() const noexcept {
    return usage_;
}

void* Buffer::mappedData() const noexcept {
    return persistentMappedData_;
}

void Buffer::write(const void* source, VkDeviceSize bytes, VkDeviceSize offset) {
    if (source == nullptr && bytes != 0)
        throw std::invalid_argument("Buffer write source must not be null");
    requireHostAccess("write");
    validateRange(bytes, offset);
    if (bytes == 0)
        return;

    void* destination = persistentMappedData_;
    bool temporaryMapping = false;
    if (destination == nullptr) {
        const VkResult result = vmaMapMemory(allocator_->nativeHandle(), allocation_, &destination);
        if (result != VK_SUCCESS)
            throw std::runtime_error("Failed to map Vulkan buffer memory");
        temporaryMapping = true;
    }

    std::memcpy(static_cast<std::byte*>(destination) + offset, source, static_cast<size_t>(bytes));
    const VkResult flushResult = vmaFlushAllocation(allocator_->nativeHandle(), allocation_, offset, bytes);
    if (temporaryMapping)
        vmaUnmapMemory(allocator_->nativeHandle(), allocation_);
    if (flushResult != VK_SUCCESS)
        throw std::runtime_error("Failed to flush Vulkan buffer memory");
}

void Buffer::flush(VkDeviceSize bytes, VkDeviceSize offset) {
    requireHostAccess("flush");
    validateRange(bytes, offset);
    if (vmaFlushAllocation(allocator_->nativeHandle(), allocation_, offset, bytes) != VK_SUCCESS)
        throw std::runtime_error("Failed to flush Vulkan buffer memory");
}

void Buffer::invalidate(VkDeviceSize bytes, VkDeviceSize offset) {
    requireHostAccess("invalidate");
    validateRange(bytes, offset);
    if (vmaInvalidateAllocation(allocator_->nativeHandle(), allocation_, offset, bytes) != VK_SUCCESS)
        throw std::runtime_error("Failed to invalidate Vulkan buffer memory");
}

void Buffer::destroy() noexcept {
    if (buffer_ != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator_->nativeHandle(), buffer_, allocation_);

    allocator_ = nullptr;
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    size_ = 0;
    usage_ = {};
    hostAccessible_ = false;
    persistentMappedData_ = nullptr;
}

void Buffer::validateRange(VkDeviceSize bytes, VkDeviceSize offset) const {
    if (!isValid())
        throw std::logic_error("Buffer is not valid");
    if (offset > size_ || (bytes != VK_WHOLE_SIZE && bytes > size_ - offset))
        throw std::out_of_range("Buffer range exceeds its size");
}

void Buffer::requireHostAccess(const char* operation) const {
    if (!hostAccessible_)
        throw std::logic_error(std::string("Cannot ") + operation + " a GPU-only buffer");
}

} // namespace vulkan_graphics
