#include "vk_descriptor.h"
#include "vk_engine.h"
#include "vk_pipeline.h"
#include "vk_shader.h"

#include <array>
#include <utility>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

int main()
{
    vk_engine::VkEngine engine{};

    vk_engine::DescriptorLayoutBuilder builder{};
    builder.AddBinding(0, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eCompute);
    vk::raii::DescriptorSetLayout descriptorLayout = builder.Build(engine.GetContext());

    vk_engine::DescriptorAllocator allocator{engine.GetContext()};
    std::array<vk::DescriptorSet, vk_engine::RenderHelper::kFramesInFlight> descriptorSets{};
    for (vk::DescriptorSet& descriptorSet : descriptorSets)
    {
        descriptorSet = allocator.Allocate(*descriptorLayout);
    }

    vk_engine::ComputePipelineDescription pipelineDescription{};
    pipelineDescription.computeShader = vk_engine::ShaderPath("clear_color.comp.spv");
    pipelineDescription.pipelineLayout.descriptorSetLayouts.push_back(*descriptorLayout);
    vk_engine::ComputePipeline pipeline(engine.GetContext(), std::move(pipelineDescription));

    vk_engine::DescriptorWriter writer{};
    engine.Run(
        [&](vk::CommandBuffer commandBuffer, vk_engine::RenderHelper& helper)
        {
            helper.TransitionToCompute();

            // This frame's fence has completed before the renderer invokes the callback.
            const vk::DescriptorSet descriptorSet = descriptorSets[helper.GetFrameIndex()];
            writer.Clear();
            writer.WriteImage(
                0, helper.GetDrawImageView(), {}, vk::ImageLayout::eGeneral, vk::DescriptorType::eStorageImage);
            writer.Update(engine.GetContext(), descriptorSet);

            pipeline.Bind(commandBuffer);
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, pipeline.GetLayout(), 0, descriptorSet, {});

            constexpr uint32_t kWorkgroupSize = 16;
            const vk::Extent2D extent = helper.GetDrawExtent();
            pipeline.Dispatch(commandBuffer,
                              (extent.width + kWorkgroupSize - 1) / kWorkgroupSize,
                              (extent.height + kWorkgroupSize - 1) / kWorkgroupSize);
        });

    return 0;
}
