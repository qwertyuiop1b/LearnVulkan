/**
 * @file ch111_async_upload.cpp
 * @brief Non-coherent staging ring upload, timeline completion and readback validation.
 */

#include <vulkan_tutorial/engine/engineering_gpu_chapter.hpp>

#include <imgui.h>
#include <cstring>
#include <iostream>

class Ch111App final : public EngineeringGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第111章：内存与异步上传"; }
    uint32_t engineeringMode() const override { return 111; }
    glm::vec4 engineeringParameters() const override {
        return {verified_ ? 1.0f : 0.0f, coherent_ ? 1.0f : 0.0f,
                static_cast<float>(budgetMiB_ / 1024.0), static_cast<float>(usageMiB_ / 1024.0)};
    }

    void onEngineeringInit() override {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physDev_, &properties);
        atomSize_ = properties.limits.nonCoherentAtomSize;
        createHostBuffer(RING_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, stagingBuffer_, stagingMemory_,
                         stagingAllocation_, coherent_);
        bool readbackCoherent = false;
        createHostBuffer(DATA_SIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT, readbackBuffer_, readbackMemory_,
                         readbackAllocation_, readbackCoherent);
        readbackCoherent_ = readbackCoherent;
        createBuffer(physDev_, device_, DATA_SIZE,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, deviceBuffer_, deviceMemory_);
        queryBudget();
        uploadAndVerify();
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Staging Ring", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            drawCapabilitySummary();
            ImGui::Text("Ring: %.1f MiB, uploaded: %.1f KiB", RING_SIZE / 1048576.0f, DATA_SIZE / 1024.0f);
            ImGui::Text("nonCoherentAtomSize: %llu", static_cast<unsigned long long>(atomSize_));
            ImGui::Text("Staging memory: %s", coherent_ ? "HOST_COHERENT fallback" : "non-coherent + flush");
            ImGui::Text("Readback: %s", readbackCoherent_ ? "coherent" : "invalidate aligned range");
            if (profile_.memoryBudget)
                ImGui::Text("Heap budget/usage: %.1f / %.1f MiB", budgetMiB_, usageMiB_);
            else
                ImGui::TextUnformatted("VK_EXT_memory_budget unavailable");
            ImGui::TextColored(verified_ ? ImVec4(0.2f, 1.0f, 0.45f, 1.0f) : ImVec4(1.0f, 0.25f, 0.2f, 1.0f),
                               "%s", verified_ ? "Async copy/readback verified" : "Upload verification failed");
        }
        ImGui::End();
    }

    void onEngineeringShutdown() override {
        vkDestroyBuffer(device_, stagingBuffer_, nullptr);
        vkFreeMemory(device_, stagingMemory_, nullptr);
        vkDestroyBuffer(device_, deviceBuffer_, nullptr);
        vkFreeMemory(device_, deviceMemory_, nullptr);
        vkDestroyBuffer(device_, readbackBuffer_, nullptr);
        vkFreeMemory(device_, readbackMemory_, nullptr);
    }

  private:
    static constexpr VkDeviceSize RING_SIZE = 4 * 1024 * 1024;
    static constexpr VkDeviceSize DATA_SIZE = 1024 * 1024;
    static constexpr uint32_t PATTERN = 0x3C7A91E5u;
    VkBuffer stagingBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory_ = VK_NULL_HANDLE;
    VkDeviceSize stagingAllocation_ = 0;
    VkBuffer deviceBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory deviceMemory_ = VK_NULL_HANDLE;
    VkBuffer readbackBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory readbackMemory_ = VK_NULL_HANDLE;
    VkDeviceSize readbackAllocation_ = 0;
    VkDeviceSize atomSize_ = 1;
    bool coherent_ = false;
    bool readbackCoherent_ = false;
    bool verified_ = false;
    double budgetMiB_ = 0.0;
    double usageMiB_ = 0.0;

    void createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer,
                          VkDeviceMemory& memory, VkDeviceSize& allocationSize, bool& coherent) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &info, nullptr, &buffer));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer, &requirements);
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(physDev_, &memoryProperties);
        uint32_t selected = UINT32_MAX;
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1u << i)) == 0) continue;
            const auto flags = memoryProperties.memoryTypes[i].propertyFlags;
            if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) continue;
            if (selected == UINT32_MAX || (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
                selected = i;
            if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) break;
        }
        if (selected == UINT32_MAX) throw std::runtime_error("No host-visible memory for upload ring");
        coherent = (memoryProperties.memoryTypes[selected].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = selected;
        VK_CHECK(vkAllocateMemory(device_, &allocation, nullptr, &memory));
        VK_CHECK(vkBindBufferMemory(device_, buffer, memory, 0));
        allocationSize = requirements.size;
    }

    void queryBudget() {
        if (!profile_.memoryBudget) return;
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        properties.pNext = &budget;
        vkGetPhysicalDeviceMemoryProperties2(physDev_, &properties);
        for (uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; ++i) {
            if (properties.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                budgetMiB_ += static_cast<double>(budget.heapBudget[i]) / 1048576.0;
                usageMiB_ += static_cast<double>(budget.heapUsage[i]) / 1048576.0;
            }
        }
    }

    void uploadAndVerify() {
        void* mapped = nullptr;
        VK_CHECK(vkMapMemory(device_, stagingMemory_, 0, DATA_SIZE, 0, &mapped));
        auto* words = static_cast<uint32_t*>(mapped);
        for (size_t i = 0; i < DATA_SIZE / sizeof(uint32_t); ++i) words[i] = PATTERN ^ static_cast<uint32_t>(i);
        if (!coherent_) {
            auto range = vulkan_tutorial::production::nonCoherentRange(
                stagingMemory_, 0, DATA_SIZE, atomSize_, stagingAllocation_);
            VK_CHECK(vkFlushMappedMemoryRanges(device_, 1, &range));
        }
        vkUnmapMemory(device_, stagingMemory_);

        VkCommandBuffer command = beginSingleTimeCommands(device_, cmdPool_);
        std::array<VkBufferCopy, 4> copies{};
        const VkDeviceSize chunk = DATA_SIZE / copies.size();
        for (size_t i = 0; i < copies.size(); ++i)
            copies[i] = {chunk * i, chunk * i, chunk};
        vkCmdCopyBuffer(command, stagingBuffer_, deviceBuffer_, static_cast<uint32_t>(copies.size()), copies.data());
        VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = deviceBuffer_;
        barrier.offset = 0;
        barrier.size = DATA_SIZE;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.bufferMemoryBarrierCount = 1;
        dependency.pBufferMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(command, &dependency);
        VkBufferCopy readback{0, 0, DATA_SIZE};
        vkCmdCopyBuffer(command, deviceBuffer_, readbackBuffer_, 1, &readback);
        VK_CHECK(vkEndCommandBuffer(command));

        VkSemaphore timeline = VK_NULL_HANDLE;
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        semaphoreInfo.pNext = &type;
        VK_CHECK(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &timeline));
        VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commandInfo.commandBuffer = command;
        VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signal.semaphore = timeline;
        signal.value = 1;
        signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &commandInfo;
        submit.signalSemaphoreInfoCount = 1;
        submit.pSignalSemaphoreInfos = &signal;
        VK_CHECK(vkQueueSubmit2(gQueue_, 1, &submit, VK_NULL_HANDLE));
        uint64_t value = 1;
        VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait.semaphoreCount = 1;
        wait.pSemaphores = &timeline;
        wait.pValues = &value;
        VK_CHECK(vkWaitSemaphores(device_, &wait, UINT64_MAX));

        VK_CHECK(vkMapMemory(device_, readbackMemory_, 0, DATA_SIZE, 0, &mapped));
        if (!readbackCoherent_) {
            auto range = vulkan_tutorial::production::nonCoherentRange(
                readbackMemory_, 0, DATA_SIZE, atomSize_, readbackAllocation_);
            VK_CHECK(vkInvalidateMappedMemoryRanges(device_, 1, &range));
        }
        words = static_cast<uint32_t*>(mapped);
        verified_ = words[0] == PATTERN && words[DATA_SIZE / sizeof(uint32_t) - 1] ==
                    (PATTERN ^ static_cast<uint32_t>(DATA_SIZE / sizeof(uint32_t) - 1));
        vkUnmapMemory(device_, readbackMemory_);
        vkDestroySemaphore(device_, timeline, nullptr);
        vkFreeCommandBuffers(device_, cmdPool_, 1, &command);
    }
};

int main() {
    try { Ch111App app; app.run("ch111 - Async Upload", 1280, 720); }
    catch (const std::exception& error) { std::cerr << "ch111 failed: " << error.what() << '\n'; return 1; }
    return 0;
}
