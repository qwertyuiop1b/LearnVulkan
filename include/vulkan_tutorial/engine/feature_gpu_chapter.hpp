#pragma once

#include <vulkan_tutorial/engine/advanced_gpu_chapter.hpp>

#include <imgui.h>

class FeatureGpuChapter : public AdvancedGpuChapterApp {
  protected:
    virtual const char* featureFragmentShader() const = 0;
    virtual const char* featureComputeShader() const = 0;
    virtual uint32_t featureMode() const = 0;
    virtual void featureParameters(AdvancedGpuPush&) const {}
    virtual void featureUi() = 0;
    virtual std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> featureStorageBufferSizes() const {
        return {1024 * sizeof(glm::vec4), 16, 16, 16};
    }
    virtual std::array<VkBufferUsageFlags, STORAGE_BUFFER_COUNT> featureExtraBufferUsage() const { return {}; }

    const char* fragmentShaderName() const final { return featureFragmentShader(); }
    const char* computeShaderName() const final { return featureComputeShader(); }
    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const final { return featureStorageBufferSizes(); }
    std::array<VkBufferUsageFlags, STORAGE_BUFFER_COUNT> extraBufferUsage() const final { return featureExtraBufferUsage(); }
    glm::uvec3 computeDispatch() const final { return {16, 1, 1}; }

    void buildChapterUi() final { featureUi(); }

    void fillPushConstants(AdvancedGpuPush& push) const final {
        push.values[0] = {static_cast<float>(extent_.width), static_cast<float>(extent_.height), elapsed_,
                          static_cast<float>(featureMode())};
        featureParameters(push);
    }

    void configureChapter() override {
        bgColor_ = {0.006f, 0.012f, 0.024f};
        interactive_.camera().setTarget({0.0f, 0.0f, -4.0f});
        interactive_.camera().setDistance(10.0f);
        interactive_.camera().setAngles(0.0f, -0.18f);
    }
};
