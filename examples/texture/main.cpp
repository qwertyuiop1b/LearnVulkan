#include "vk_buffer.h"
#include "vk_descriptor.h"
#include "vk_engine.h"
#include "vk_imgui.h"
#include "vk_pipeline.h"
#include "vk_shader.h"
#include "vk_texture.h"
#include <array>
#include <filesystem>
#include <span>
#include <utility>
#include <glm/glm.hpp>
#include "imgui.h"

struct Vertex
{
    glm::vec2 position;
    glm::vec2 texCoord;
    static vk_engine::VertexInputDescription GetInputDescription()
    {
        vk_engine::VertexInputDescription description;
        description.bindings.push_back(
            vk::VertexInputBindingDescription{0, sizeof(Vertex), vk::VertexInputRate::eVertex});
        description.attributes.push_back(
            vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, position)});
        description.attributes.push_back(
            vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)});
        return description;
    }
};

namespace
{
std::filesystem::path AssetPath(const char* name)
{
#ifdef VK_ENGINE_TEXTURE_ASSET_DIR
    return std::filesystem::path{VK_ENGINE_TEXTURE_ASSET_DIR} / name;
#else
    return std::filesystem::path{"assets"} / name;
#endif
}
} // namespace
int main()
{
    vk_engine::VkEngine engine{};
    const std::array<Vertex, 4> vertices{Vertex{{-0.75F, -0.75F}, {0.0F, 1.0F}},
                                         Vertex{{0.75F, -0.75F}, {1.0F, 1.0F}},
                                         Vertex{{0.75F, 0.75F}, {1.0F, 0.0F}},
                                         Vertex{{-0.75F, 0.75F}, {0.0F, 0.0F}}};
    const std::array<uint32_t, 6> indices{0, 1, 2, 2, 3, 0};
    vk_engine::Buffer vertexBuffer(engine.GetContext(),
                                   sizeof(vertices),
                                   vk::BufferUsageFlagBits::eVertexBuffer,
                                   vk::MemoryPropertyFlagBits::eHostVisible |
                                       vk::MemoryPropertyFlagBits::eHostCoherent);
    vertexBuffer.Write(std::as_bytes(std::span<const Vertex>{vertices}));
    vk_engine::Buffer indexBuffer(engine.GetContext(),
                                  sizeof(indices),
                                  vk::BufferUsageFlagBits::eIndexBuffer,
                                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    indexBuffer.Write(std::as_bytes(std::span<const uint32_t>{indices}));
    vk_engine::VkTexture texture(engine.GetContext(), AssetPath("awesome_face.png"));
    vk_engine::DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.AddBinding(0, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment);
    vk::raii::DescriptorSetLayout setLayout = layoutBuilder.Build(engine.GetContext());
    vk_engine::DescriptorAllocator descriptorAllocator(engine.GetContext());
    const vk::DescriptorSet descriptorSet = descriptorAllocator.Allocate(*setLayout);
    const vk::DescriptorImageInfo imageInfo = texture.GetDescriptorInfo();
    vk_engine::DescriptorWriter writer;
    writer.WriteImage(
        0, imageInfo.imageView, imageInfo.sampler, imageInfo.imageLayout, vk::DescriptorType::eCombinedImageSampler);
    writer.Update(engine.GetContext(), descriptorSet);
    vk_engine::GraphicsPipelineDescription pipelineDescription{};
    pipelineDescription.vertexShader = vk_engine::ShaderPath("texture.vert.spv");
    pipelineDescription.fragmentShader = vk_engine::ShaderPath("texture.frag.spv");
    pipelineDescription.vertexInput = Vertex::GetInputDescription();
    pipelineDescription.pipelineLayout.descriptorSetLayouts.push_back(*setLayout);
    vk_engine::GraphicsPipeline pipeline(
        engine.GetContext(), std::move(pipelineDescription), engine.GetSwapchain().GetImageFormat());
    vk_engine::VkImGui imgui(engine.GetContext(),
                             &engine.GetWindowHandle(),
                             engine.GetDrawImageFormat(),
                             2,
                             engine.GetSwapchain().GetImageCount());
    bool showDemoWindow = true;
    engine.Run(
        [&](vk::CommandBuffer commandBuffer, vk_engine::RenderHelper& helper)
        {
            imgui.BeginFrame();
            ImGui::Begin("Texture Example");
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Text("Renders a texture-mapped quad.");
            ImGui::Checkbox("Show Demo Window", &showDemoWindow);
            ImGui::End();
            if (showDemoWindow)
            {
                ImGui::ShowDemoWindow(&showDemoWindow);
            }

            helper.TransitionToGraphics();
            const vk::Format colorFormat = helper.GetDrawImageFormat();
            const vk::Extent2D extent = helper.GetDrawExtent();
            pipeline.EnsureCompatible(colorFormat);
            vk::RenderingAttachmentInfo colorAttachment{};
            colorAttachment.setImageView(helper.GetDrawImageView())
                .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setClearValue(vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.03F, 0.03F, 0.03F, 1.0F}}});
            vk::RenderingInfo renderingInfo{};
            renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, extent})
                .setLayerCount(1)
                .setColorAttachments(colorAttachment);
            commandBuffer.beginRendering(renderingInfo);
            pipeline.Bind(commandBuffer);
            commandBuffer.setViewport(
                0,
                vk::Viewport{
                    0.0F, 0.0F, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0F, 1.0F});
            commandBuffer.setScissor(0, vk::Rect2D{{0, 0}, extent});
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics, pipeline.GetLayout(), 0, descriptorSet, {});
            commandBuffer.bindVertexBuffers(0, vertexBuffer.GetHandle(), vk::DeviceSize{0});
            commandBuffer.bindIndexBuffer(indexBuffer.GetHandle(), 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(6, 1, 0, 0, 0);
            imgui.Render(commandBuffer);
            commandBuffer.endRendering();
        });
}
