/**
 * @file ch109_sync2_multiqueue.cpp
 * @brief Submit2 + timeline + queue-family release/acquire with readback verification.
 */

#include <vulkan_tutorial/engine/engineering_gpu_chapter.hpp>

#include <imgui.h>
#include <iostream>

class Ch109App final : public EngineeringGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第109章：Sync2 与多队列"; }
    uint32_t engineeringMode() const override { return 109; }
    bool chapterUsesSynchronization2() const override { return true; }
    glm::vec4 engineeringParameters() const override {
        return {verified_ ? 1.0f : 0.0f, dedicatedQueue_ ? 1.0f : 0.0f,
                static_cast<float>(timelineValue_), 0.0f};
    }

    void onEngineeringInit() override {
        if (!profile_.synchronization2 || !profile_.timelineSemaphore)
            return;
        computeFamily_ = qIdx_.computeFamily.value_or(qIdx_.graphicsFamily.value());
        graphicsFamily_ = qIdx_.graphicsFamily.value();
        dedicatedQueue_ = computeFamily_ != graphicsFamily_;
        vkGetDeviceQueue(device_, computeFamily_, 0, &computeQueue_);

        createBuffer(physDev_, device_, sizeof(uint32_t) * WORD_COUNT,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sharedBuffer_, sharedMemory_);
        createBuffer(physDev_, device_, sizeof(uint32_t) * WORD_COUNT, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     readbackBuffer_, readbackMemory_);
        createTimeline();
        createCommands();
        executeOwnershipTransfer();
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Submit2 / Ownership", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            drawCapabilitySummary();
            ImGui::Text("Synchronization2: %s", profile_.synchronization2 ? "enabled" : "unavailable");
            ImGui::Text("Timeline semaphore: %s", profile_.timelineSemaphore ? "enabled" : "unavailable");
            ImGui::Text("Compute family: %u  Graphics family: %u", computeFamily_, graphicsFamily_);
            ImGui::Text("Queue ownership: %s", dedicatedQueue_ ? "release/acquire" : "same-family fallback");
            ImGui::Text("Timeline completed: %llu", static_cast<unsigned long long>(timelineValue_));
            ImGui::TextColored(verified_ ? ImVec4(0.2f, 1.0f, 0.45f, 1.0f) : ImVec4(1.0f, 0.3f, 0.2f, 1.0f),
                               "%s", verified_ ? "GPU readback verified" : "GPU path unavailable/failed");
        }
        ImGui::End();
    }

    void onEngineeringShutdown() override {
        if (device_ == VK_NULL_HANDLE) return;
        vkDestroyCommandPool(device_, computePool_, nullptr);
        vkDestroyCommandPool(device_, graphicsPool_, nullptr);
        vkDestroySemaphore(device_, timeline_, nullptr);
        vkDestroyBuffer(device_, sharedBuffer_, nullptr);
        vkFreeMemory(device_, sharedMemory_, nullptr);
        vkDestroyBuffer(device_, readbackBuffer_, nullptr);
        vkFreeMemory(device_, readbackMemory_, nullptr);
    }

  private:
    static constexpr uint32_t WORD_COUNT = 1024;
    static constexpr uint32_t PATTERN = 0xA5A5A5A5u;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    VkCommandPool computePool_ = VK_NULL_HANDLE;
    VkCommandPool graphicsPool_ = VK_NULL_HANDLE;
    VkCommandBuffer computeCmd_ = VK_NULL_HANDLE;
    VkCommandBuffer graphicsCmd_ = VK_NULL_HANDLE;
    VkBuffer sharedBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory sharedMemory_ = VK_NULL_HANDLE;
    VkBuffer readbackBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory readbackMemory_ = VK_NULL_HANDLE;
    uint32_t computeFamily_ = 0;
    uint32_t graphicsFamily_ = 0;
    uint64_t timelineValue_ = 0;
    bool dedicatedQueue_ = false;
    bool verified_ = false;

    void createTimeline() {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        VK_CHECK(vkCreateSemaphore(device_, &info, nullptr, &timeline_));
    }

    void allocateCommand(uint32_t family, VkCommandPool& pool, VkCommandBuffer& command) {
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.queueFamilyIndex = family;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &pool));
        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device_, &alloc, &command));
    }

    void createCommands() {
        allocateCommand(computeFamily_, computePool_, computeCmd_);
        allocateCommand(graphicsFamily_, graphicsPool_, graphicsCmd_);

        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(computeCmd_, &begin));
        vkCmdFillBuffer(computeCmd_, sharedBuffer_, 0, VK_WHOLE_SIZE, PATTERN);
        VkBufferMemoryBarrier2 release{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        release.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        release.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        release.dstStageMask = dedicatedQueue_ ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        release.dstAccessMask = dedicatedQueue_ ? VK_ACCESS_2_NONE : VK_ACCESS_2_TRANSFER_READ_BIT;
        release.srcQueueFamilyIndex = dedicatedQueue_ ? computeFamily_ : VK_QUEUE_FAMILY_IGNORED;
        release.dstQueueFamilyIndex = dedicatedQueue_ ? graphicsFamily_ : VK_QUEUE_FAMILY_IGNORED;
        release.buffer = sharedBuffer_;
        release.offset = 0;
        release.size = VK_WHOLE_SIZE;
        VkDependencyInfo releaseDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        releaseDependency.bufferMemoryBarrierCount = 1;
        releaseDependency.pBufferMemoryBarriers = &release;
        vkCmdPipelineBarrier2(computeCmd_, &releaseDependency);
        VK_CHECK(vkEndCommandBuffer(computeCmd_));

        VK_CHECK(vkBeginCommandBuffer(graphicsCmd_, &begin));
        if (dedicatedQueue_) {
            VkBufferMemoryBarrier2 acquire{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
            acquire.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
            acquire.srcAccessMask = VK_ACCESS_2_NONE;
            acquire.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            acquire.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            acquire.srcQueueFamilyIndex = computeFamily_;
            acquire.dstQueueFamilyIndex = graphicsFamily_;
            acquire.buffer = sharedBuffer_;
            acquire.offset = 0;
            acquire.size = VK_WHOLE_SIZE;
            VkDependencyInfo acquireDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            acquireDependency.bufferMemoryBarrierCount = 1;
            acquireDependency.pBufferMemoryBarriers = &acquire;
            vkCmdPipelineBarrier2(graphicsCmd_, &acquireDependency);
        }
        VkBufferCopy copy{0, 0, sizeof(uint32_t) * WORD_COUNT};
        vkCmdCopyBuffer(graphicsCmd_, sharedBuffer_, readbackBuffer_, 1, &copy);
        VK_CHECK(vkEndCommandBuffer(graphicsCmd_));
    }

    void executeOwnershipTransfer() {
        VkCommandBufferSubmitInfo computeCommand{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        computeCommand.commandBuffer = computeCmd_;
        VkSemaphoreSubmitInfo computeSignal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        computeSignal.semaphore = timeline_;
        computeSignal.value = 1;
        computeSignal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSubmitInfo2 computeSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        computeSubmit.commandBufferInfoCount = 1;
        computeSubmit.pCommandBufferInfos = &computeCommand;
        computeSubmit.signalSemaphoreInfoCount = 1;
        computeSubmit.pSignalSemaphoreInfos = &computeSignal;
        VK_CHECK(vkQueueSubmit2(computeQueue_, 1, &computeSubmit, VK_NULL_HANDLE));

        VkSemaphoreSubmitInfo graphicsWait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        graphicsWait.semaphore = timeline_;
        graphicsWait.value = 1;
        graphicsWait.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        VkCommandBufferSubmitInfo graphicsCommand{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        graphicsCommand.commandBuffer = graphicsCmd_;
        VkSemaphoreSubmitInfo graphicsSignal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        graphicsSignal.semaphore = timeline_;
        graphicsSignal.value = 2;
        graphicsSignal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSubmitInfo2 graphicsSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        graphicsSubmit.waitSemaphoreInfoCount = 1;
        graphicsSubmit.pWaitSemaphoreInfos = &graphicsWait;
        graphicsSubmit.commandBufferInfoCount = 1;
        graphicsSubmit.pCommandBufferInfos = &graphicsCommand;
        graphicsSubmit.signalSemaphoreInfoCount = 1;
        graphicsSubmit.pSignalSemaphoreInfos = &graphicsSignal;
        VK_CHECK(vkQueueSubmit2(gQueue_, 1, &graphicsSubmit, VK_NULL_HANDLE));

        uint64_t waitValue = 2;
        VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait.semaphoreCount = 1;
        wait.pSemaphores = &timeline_;
        wait.pValues = &waitValue;
        VK_CHECK(vkWaitSemaphores(device_, &wait, UINT64_MAX));
        VK_CHECK(vkGetSemaphoreCounterValue(device_, timeline_, &timelineValue_));
        void* mapped = nullptr;
        VK_CHECK(vkMapMemory(device_, readbackMemory_, 0, sizeof(uint32_t), 0, &mapped));
        verified_ = *static_cast<const uint32_t*>(mapped) == PATTERN;
        vkUnmapMemory(device_, readbackMemory_);
    }
};

int main() {
    try { Ch109App app; app.run("ch109 - Sync2 Multi Queue", 1280, 720); }
    catch (const std::exception& error) { std::cerr << "ch109 failed: " << error.what() << '\n'; return 1; }
    return 0;
}
