/**
 * @file ch101_procedural_heightmap.cpp
 * @brief GPU procedural heightmap generation and 3D height-field rendering.
 */

#include <vulkan_tutorial/engine/demo_app.hpp>

#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace vulkan_tutorial;

namespace {
constexpr uint32_t HEIGHTMAP_SIZE = 512;
constexpr VkFormat HEIGHTMAP_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;

struct alignas(16) HeightPush {
    float time;
    float seed;
    float frequency;
    float amplitude;
    int32_t octaves;
    float ridgeWeight;
    float warpStrength;
    float padding;
};

struct TerrainPush {
    float time;
    float aspect;
    float heightScale;
    float terrainScale;
};

#ifdef CH103_GPU_CLIPMAP_TERRAIN
struct alignas(16) ClipmapPush {
    glm::mat4 viewProjection;
    glm::vec4 cameraPosition;
    glm::vec4 terrainParams; // heightScale, terrainScale, baseCellSize, gridResolution
};
static_assert(sizeof(ClipmapPush) == 96);
#endif

#ifdef CH102_HYDRAULIC_EROSION
struct alignas(16) ErosionPush {
    float erosionRate;
    float depositionRate;
    float evaporation;
    float talus;
};
#endif

static_assert(sizeof(HeightPush) == 32);
static_assert(sizeof(TerrainPush) == 16);
#ifdef CH102_HYDRAULIC_EROSION
static_assert(sizeof(ErosionPush) == 16);
#endif
} // namespace

class Ch101App final : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.08f, 0.13f, 0.20f};
        createHeightmap();
        createDescriptors();
        createComputePipeline();
#ifdef CH102_HYDRAULIC_EROSION
        createErosionPipeline();
#endif
        createTerrainPipeline();
#ifdef CH103_GPU_CLIPMAP_TERRAIN
        interactive_.camera().setTarget({0.0f, 4.0f, 0.0f});
        interactive_.camera().setDistance(28.0f);
        interactive_.camera().setAngles(45.0f, 28.0f);
#endif
    }

#ifdef CH103_GPU_CLIPMAP_TERRAIN
    bool needsDepthBuffer() const override { return true; }
#endif

    void onUpdate() override {
        elapsed_ += 0.016f;
    }

    void buildUi() override {
#ifdef CH104_PROCEDURAL_BIOMES
        interactive_.buildDebugPanel("第104章：GPU 程序化生物群系");
#elif defined(CH103_GPU_CLIPMAP_TERRAIN)
        interactive_.buildDebugPanel("第103章：GPU Clipmap 地形");
#elif defined(CH102_HYDRAULIC_EROSION)
        interactive_.buildDebugPanel("第102章：GPU 地形侵蚀");
#else
        interactive_.buildDebugPanel("第101章：GPU 程序化高度图");
#endif
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(330, 0), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Terrain Generator", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderFloat("Seed", &seed_, 0.0f, 1000.0f, "%.1f");
            ImGui::SliderFloat("Frequency", &frequency_, 1.5f, 12.0f, "%.2f");
            ImGui::SliderFloat("Amplitude", &amplitude_, 0.15f, 0.8f, "%.2f");
            ImGui::SliderInt("Octaves", &octaves_, 1, 9);
            ImGui::SliderFloat("Ridged", &ridgeWeight_, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Domain Warp", &warpStrength_, 0.0f, 2.5f, "%.2f");
            ImGui::SliderFloat("Height", &heightScale_, 3.0f, 16.0f, "%.1f m");
            ImGui::Checkbox("Animate Noise", &animateNoise_);
#ifdef CH103_GPU_CLIPMAP_TERRAIN
            ImGui::Separator();
            ImGui::SliderFloat("Base Cell Size", &baseCellSize_, 0.15f, 1.5f, "%.2f m");
            ImGui::Text("Clipmap: 5 levels x 64x64 cells");
            ImGui::Text("Draw vertices: %u", (64u - 1u) * (64u - 1u) * 6u * 5u);
#ifdef CH104_PROCEDURAL_BIOMES
            ImGui::Text("Biomes: desert / grassland / forest / rock / snow");
#endif
#endif
#ifdef CH102_HYDRAULIC_EROSION
            ImGui::Separator();
            ImGui::SliderInt("Erosion Iterations", &erosionIterations_, 2, 32);
            erosionIterations_ += erosionIterations_ & 1; // Final result remains in the primary image.
            ImGui::SliderFloat("Erosion Rate", &erosionRate_, 0.01f, 0.35f, "%.3f");
            ImGui::SliderFloat("Deposition", &depositionRate_, 0.01f, 0.45f, "%.3f");
            ImGui::SliderFloat("Evaporation", &evaporation_, 0.0f, 0.12f, "%.3f");
            ImGui::SliderFloat("Talus", &talus_, 0.001f, 0.08f, "%.3f");
#endif
            ImGui::Text("Heightmap: %ux%u RGBA16F", HEIGHTMAP_SIZE, HEIGHTMAP_SIZE);
            ImGui::Text("Compute groups: %ux%u", HEIGHTMAP_SIZE / 16, HEIGHTMAP_SIZE / 16);
        }
        ImGui::End();
    }

    void onRecordPreRender(VkCommandBuffer cmd, uint32_t) override {
#ifdef CH102_HYDRAULIC_EROSION
        std::array<VkImageMemoryBarrier, 2> before{};
        VkImage images[] = {heightmap_, erosionImage_};
        for (size_t i = 0; i < before.size(); ++i) {
            before[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            before[i].oldLayout = imageInitialized_ ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
            before[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            before[i].srcAccessMask = imageInitialized_ ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT : 0;
            before[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            before[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            before[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            before[i].image = images[i];
            before[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        VkPipelineStageFlags sourceStages = imageInitialized_
                                                ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                                : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        vkCmdPipelineBarrier(cmd, sourceStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                             static_cast<uint32_t>(before.size()), before.data());
#else
        VkImageMemoryBarrier before{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        before.oldLayout = imageInitialized_ ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
        before.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.image = heightmap_;
        before.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        before.srcAccessMask = imageInitialized_ ? VK_ACCESS_SHADER_READ_BIT : 0;
        before.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             imageInitialized_ ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &before);
#endif
        imageInitialized_ = true;

        HeightPush push{};
        push.time = animateNoise_ ? elapsed_ : 0.0f;
        push.seed = seed_;
        push.frequency = frequency_;
        push.amplitude = amplitude_;
        push.octaves = octaves_;
        push.ridgeWeight = ridgeWeight_;
        push.warpStrength = warpStrength_;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout_, 0, 1, &computeSet_, 0, nullptr);
        vkCmdPushConstants(cmd, computeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, HEIGHTMAP_SIZE / 16, HEIGHTMAP_SIZE / 16, 1);

#ifdef CH102_HYDRAULIC_EROSION
        VkImageMemoryBarrier iterationBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        iterationBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        iterationBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        iterationBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        iterationBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        iterationBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        iterationBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        iterationBarrier.image = heightmap_;
        iterationBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &iterationBarrier);

        ErosionPush erosionPush{erosionRate_, depositionRate_, evaporation_, talus_};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, erosionPipeline_);
        for (int iteration = 0; iteration < erosionIterations_; ++iteration) {
            const uint32_t setIndex = uint32_t(iteration & 1);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, erosionLayout_, 0, 1,
                                    &erosionSets_[setIndex], 0, nullptr);
            vkCmdPushConstants(cmd, erosionLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(erosionPush), &erosionPush);
            vkCmdDispatch(cmd, HEIGHTMAP_SIZE / 16, HEIGHTMAP_SIZE / 16, 1);

            iterationBarrier.image = setIndex == 0 ? erosionImage_ : heightmap_;
            const bool finalIteration = iteration + 1 == erosionIterations_;
            iterationBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 finalIteration ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                                : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &iterationBarrier);
        }
#else
        VkImageMemoryBarrier after{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        after.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        after.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        after.image = heightmap_;
        after.subresourceRange = before.subresourceRange;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &after);
#endif
    }

    void onRecordRender(VkCommandBuffer cmd, uint32_t) override {
        VkViewport viewport{0, 0, float(extent_.width), float(extent_.height), 0, 1};
        VkRect2D scissor{{0, 0}, extent_};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrainPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrainLayout_, 0, 1, &terrainSet_, 0, nullptr);
#ifdef CH103_GPU_CLIPMAP_TERRAIN
        ClipmapPush push{};
        const float aspect = float(extent_.width) / float(std::max(1u, extent_.height));
        push.viewProjection = interactive_.camera().projectionMatrix(aspect, 55.0f, 0.1f, 1200.0f) *
                              interactive_.camera().viewMatrix();
        push.cameraPosition = glm::vec4(interactive_.camera().eyePosition(), 1.0f);
        push.terrainParams = glm::vec4(heightScale_, 256.0f, baseCellSize_, 64.0f);
        vkCmdPushConstants(cmd, terrainLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(push), &push);
        constexpr uint32_t verticesPerLevel = (64 - 1) * (64 - 1) * 6;
        vkCmdDraw(cmd, verticesPerLevel, 5, 0, 0);
#else
        TerrainPush push{elapsed_, float(extent_.width) / float(std::max(1u, extent_.height)), heightScale_, 42.0f};
        vkCmdPushConstants(cmd, terrainLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(cmd, 3, 1, 0, 0);
#endif
    }

    void onShutdown() override {
        vkDestroyPipeline(device_, terrainPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, terrainLayout_, nullptr);
        vkDestroyPipeline(device_, computePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, computeLayout_, nullptr);
#ifdef CH102_HYDRAULIC_EROSION
        vkDestroyPipeline(device_, erosionPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, erosionLayout_, nullptr);
#endif
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, terrainSetLayout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, computeSetLayout_, nullptr);
#ifdef CH102_HYDRAULIC_EROSION
        vkDestroyDescriptorSetLayout(device_, erosionSetLayout_, nullptr);
        vkDestroyImageView(device_, erosionView_, nullptr);
        vkDestroyImage(device_, erosionImage_, nullptr);
        vkFreeMemory(device_, erosionMemory_, nullptr);
#endif
        vkDestroySampler(device_, heightmapSampler_, nullptr);
        vkDestroyImageView(device_, heightmapView_, nullptr);
        vkDestroyImage(device_, heightmap_, nullptr);
        vkFreeMemory(device_, heightmapMemory_, nullptr);
    }

  private:
    VkImage heightmap_ = VK_NULL_HANDLE;
    VkDeviceMemory heightmapMemory_ = VK_NULL_HANDLE;
    VkImageView heightmapView_ = VK_NULL_HANDLE;
    VkSampler heightmapSampler_ = VK_NULL_HANDLE;
#ifdef CH102_HYDRAULIC_EROSION
    VkImage erosionImage_ = VK_NULL_HANDLE;
    VkDeviceMemory erosionMemory_ = VK_NULL_HANDLE;
    VkImageView erosionView_ = VK_NULL_HANDLE;
#endif
    VkDescriptorSetLayout computeSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout terrainSetLayout_ = VK_NULL_HANDLE;
#ifdef CH102_HYDRAULIC_EROSION
    VkDescriptorSetLayout erosionSetLayout_ = VK_NULL_HANDLE;
#endif
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet computeSet_ = VK_NULL_HANDLE;
    VkDescriptorSet terrainSet_ = VK_NULL_HANDLE;
#ifdef CH102_HYDRAULIC_EROSION
    std::array<VkDescriptorSet, 2> erosionSets_{};
#endif
    VkPipelineLayout computeLayout_ = VK_NULL_HANDLE;
    VkPipeline computePipeline_ = VK_NULL_HANDLE;
#ifdef CH102_HYDRAULIC_EROSION
    VkPipelineLayout erosionLayout_ = VK_NULL_HANDLE;
    VkPipeline erosionPipeline_ = VK_NULL_HANDLE;
#endif
    VkPipelineLayout terrainLayout_ = VK_NULL_HANDLE;
    VkPipeline terrainPipeline_ = VK_NULL_HANDLE;
    bool imageInitialized_ = false;
    bool animateNoise_ = false;
    float elapsed_ = 0.0f;
    float seed_ = 73.0f;
    float frequency_ = 5.8f;
    float amplitude_ = 0.52f;
    float ridgeWeight_ = 0.38f;
    float warpStrength_ = 1.15f;
    float heightScale_ = 10.0f;
    int octaves_ = 7;
#ifdef CH103_GPU_CLIPMAP_TERRAIN
    float baseCellSize_ = 0.5f;
#endif
#ifdef CH102_HYDRAULIC_EROSION
    int erosionIterations_ = 12;
    float erosionRate_ = 0.16f;
    float depositionRate_ = 0.20f;
    float evaporation_ = 0.025f;
    float talus_ = 0.018f;
#endif

    void createHeightmap() {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physDev_, HEIGHTMAP_FORMAT, &properties);
        const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & required) != required)
            throw std::runtime_error("RGBA16F storage/sampled images are unsupported on this device");

        createImage(physDev_, device_, HEIGHTMAP_SIZE, HEIGHTMAP_SIZE, HEIGHTMAP_FORMAT, VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    heightmap_, heightmapMemory_);

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = heightmap_;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = HEIGHTMAP_FORMAT;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &viewInfo, nullptr, &heightmapView_));

#ifdef CH102_HYDRAULIC_EROSION
        createImage(physDev_, device_, HEIGHTMAP_SIZE, HEIGHTMAP_SIZE, HEIGHTMAP_FORMAT, VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    erosionImage_, erosionMemory_);
        viewInfo.image = erosionImage_;
        VK_CHECK(vkCreateImageView(device_, &viewInfo, nullptr, &erosionView_));
#endif

        VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        const bool linear = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
        samplerInfo.magFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        samplerInfo.minFilter = samplerInfo.magFilter;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.maxLod = 0.0f;
        VK_CHECK(vkCreateSampler(device_, &samplerInfo, nullptr, &heightmapSampler_));
    }

    void createDescriptors() {
        VkDescriptorSetLayoutBinding storageBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                                     VK_SHADER_STAGE_COMPUTE_BIT};
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &storageBinding;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &computeSetLayout_));

        VkDescriptorSetLayoutBinding sampledBinding{
            0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
#ifdef CH103_GPU_CLIPMAP_TERRAIN
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
#else
            VK_SHADER_STAGE_FRAGMENT_BIT
#endif
        };
        layoutInfo.pBindings = &sampledBinding;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &terrainSetLayout_));

#ifdef CH102_HYDRAULIC_EROSION
        std::array<VkDescriptorSetLayoutBinding, 2> erosionBindings = {{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        }};
        layoutInfo.bindingCount = static_cast<uint32_t>(erosionBindings.size());
        layoutInfo.pBindings = erosionBindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &erosionSetLayout_));

        std::array<VkDescriptorPoolSize, 2> poolSizes = {{{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5},
                                                          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}}};
#else
        std::array<VkDescriptorPoolSize, 2> poolSizes = {{{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                                                          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}}};
#endif
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
#ifdef CH102_HYDRAULIC_EROSION
        poolInfo.maxSets = 4;
#else
        poolInfo.maxSets = 2;
#endif
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));

#ifdef CH102_HYDRAULIC_EROSION
        VkDescriptorSetLayout layouts[] = {computeSetLayout_, terrainSetLayout_, erosionSetLayout_, erosionSetLayout_};
        VkDescriptorSet sets[] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
#else
        VkDescriptorSetLayout layouts[] = {computeSetLayout_, terrainSetLayout_};
        VkDescriptorSet sets[] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
#endif
        VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocateInfo.descriptorPool = descriptorPool_;
        allocateInfo.descriptorSetCount = static_cast<uint32_t>(std::size(layouts));
        allocateInfo.pSetLayouts = layouts;
        VK_CHECK(vkAllocateDescriptorSets(device_, &allocateInfo, sets));
        computeSet_ = sets[0];
        terrainSet_ = sets[1];
#ifdef CH102_HYDRAULIC_EROSION
        erosionSets_[0] = sets[2];
        erosionSets_[1] = sets[3];
#endif

        VkDescriptorImageInfo storageInfo{VK_NULL_HANDLE, heightmapView_, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo sampledInfo{heightmapSampler_, heightmapView_, VK_IMAGE_LAYOUT_GENERAL};
        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = computeSet_;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &storageInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = terrainSet_;
        writes[1].dstBinding = 0;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &sampledInfo;
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

#ifdef CH102_HYDRAULIC_EROSION
        VkDescriptorImageInfo primaryStorage{VK_NULL_HANDLE, heightmapView_, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo secondaryStorage{VK_NULL_HANDLE, erosionView_, VK_IMAGE_LAYOUT_GENERAL};
        std::array<VkWriteDescriptorSet, 4> erosionWrites{};
        VkDescriptorImageInfo* erosionInfos[] = {&primaryStorage, &secondaryStorage, &secondaryStorage, &primaryStorage};
        for (uint32_t i = 0; i < erosionWrites.size(); ++i) {
            erosionWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            erosionWrites[i].dstSet = erosionSets_[i / 2];
            erosionWrites[i].dstBinding = i % 2;
            erosionWrites[i].descriptorCount = 1;
            erosionWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            erosionWrites[i].pImageInfo = erosionInfos[i];
        }
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(erosionWrites.size()), erosionWrites.data(), 0, nullptr);
#endif
    }

    void createComputePipeline() {
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HeightPush)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &computeSetLayout_;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &computeLayout_));

        VkShaderModule module = createShaderModuleFromFile(device_, "procedural_heightmap.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                              VK_SHADER_STAGE_COMPUTE_BIT, module, "main"};
        pipelineInfo.layout = computeLayout_;
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline_));
        vkDestroyShaderModule(device_, module, nullptr);
    }

#ifdef CH102_HYDRAULIC_EROSION
    void createErosionPipeline() {
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ErosionPush)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &erosionSetLayout_;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &erosionLayout_));

        VkShaderModule module = createShaderModuleFromFile(device_, "terrain_erosion.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                              VK_SHADER_STAGE_COMPUTE_BIT, module, "main"};
        pipelineInfo.layout = erosionLayout_;
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &erosionPipeline_));
        vkDestroyShaderModule(device_, module, nullptr);
    }
#endif

    void createTerrainPipeline() {
#ifdef CH103_GPU_CLIPMAP_TERRAIN
        VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ClipmapPush)};
#else
        VkPushConstantRange range{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(TerrainPush)};
#endif
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &terrainSetLayout_;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &terrainLayout_));

#ifdef CH103_GPU_CLIPMAP_TERRAIN
        VkShaderModule vert = createShaderModuleFromFile(device_, "clipmap_terrain.vert.spv");
#ifdef CH104_PROCEDURAL_BIOMES
        VkShaderModule frag = createShaderModuleFromFile(device_, "biome_terrain.frag.spv");
#else
        VkShaderModule frag = createShaderModuleFromFile(device_, "clipmap_terrain.frag.spv");
#endif
#else
        VkShaderModule vert = createShaderModuleFromFile(device_, "procedural_terrain.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "procedural_terrain.frag.spv");
#endif
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = {{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vert, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main"},
        }};
        VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
#ifdef CH103_GPU_CLIPMAP_TERRAIN
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;
#endif
        VkPipelineColorBlendAttachmentState blend{};
        blend.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blending.attachmentCount = 1;
        blending.pAttachments = &blend;
        std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();

        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &assembly;
        pipelineInfo.pViewportState = &viewport;
        pipelineInfo.pRasterizationState = &raster;
        pipelineInfo.pMultisampleState = &multisample;
#ifdef CH103_GPU_CLIPMAP_TERRAIN
        pipelineInfo.pDepthStencilState = &depth;
#endif
        pipelineInfo.pColorBlendState = &blending;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = terrainLayout_;
        pipelineInfo.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &terrainPipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }
};

int main() {
    try {
        Ch101App app;
#ifdef CH104_PROCEDURAL_BIOMES
        app.run("ch104 - GPU Procedural Biomes", 1280, 720);
#elif defined(CH103_GPU_CLIPMAP_TERRAIN)
        app.run("ch103 - GPU Clipmap Terrain", 1280, 720);
#elif defined(CH102_HYDRAULIC_EROSION)
        app.run("ch102 - GPU Hydraulic Erosion", 1280, 720);
#else
        app.run("ch101 - GPU Procedural Heightmap", 1280, 720);
#endif
    } catch (const std::exception& error) {
#ifdef CH104_PROCEDURAL_BIOMES
        std::cerr << "ch104 failed: " << error.what() << '\n';
#elif defined(CH103_GPU_CLIPMAP_TERRAIN)
        std::cerr << "ch103 failed: " << error.what() << '\n';
#elif defined(CH102_HYDRAULIC_EROSION)
        std::cerr << "ch102 failed: " << error.what() << '\n';
#else
        std::cerr << "ch101 failed: " << error.what() << '\n';
#endif
        return 1;
    }
    return 0;
}
