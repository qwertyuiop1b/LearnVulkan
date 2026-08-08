#include "texture_vertex.h"
#include "vk_buffer.h"
#include "vk_descriptor.h"
#include "vk_engine.h"
#include "vk_pipeline.h"
#include "vk_texture.h"
#include <array>
#include <filesystem>
#include <span>
#include <utility>
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
std::filesystem::path ShaderPath(const char* name)
{
#ifdef VK_ENGINE_TEXTURE_SHADER_DIR
    return std::filesystem::path{VK_ENGINE_TEXTURE_SHADER_DIR} / name;
#else
    return std::filesystem::path{"shaders"} / name;
#endif
}
} // namespace
int main()
{
    vk_engine::VkEngine engine{};
    const std::array<texture_example::Vertex, 4> vertices{texture_example::Vertex{{-0.75F, -0.75F}, {0.0F, 1.0F}},
                                                          texture_example::Vertex{{0.75F, -0.75F}, {1.0F, 1.0F}},
                                                          texture_example::Vertex{{0.75F, 0.75F}, {1.0F, 0.0F}},
                                                          texture_example::Vertex{{-0.75F, 0.75F}, {0.0F, 0.0F}}};
    const std::array<uint32_t, 6> indices{0, 1, 2, 2, 3, 0};
    vk_engine::Buffer vertexBuffer(engine.GetContext(),
                                     sizeof(vertices),
                                     vk::BufferUsageFlagBits::eVertexBuffer,
                                     vk::MemoryPropertyFlagBits::eHostVisible |
                                         vk::MemoryPropertyFlagBits::eHostCoherent);
    vertexBuffer.Write(std::as_bytes(std::span<const texture_example::Vertex>{vertices}));
    vk_engine::Buffer indexBuffer(engine.GetContext(),
                                    sizeof(indices),
                                    vk::BufferUsageFlagBits::eIndexBuffer,
                                    vk::MemoryPropertyFlagBits::eHostVisible |
                                        vk::MemoryPropertyFlagBits::eHostCoherent);
    indexBuffer.Write(std::as_bytes(std::span<const uint32_t>{indices}));
    vk_engine::VkTexture texture(engine.GetContext(), AssetPath("awesome_face.png"));
    vk_engine::VkTextureDescriptor descriptor(engine.GetContext(), texture);
    vk_engine::GraphicsPipelineDescription pipelineDescription{};
    pipelineDescription.vertexShader = ShaderPath("texture.vert.spv");
    pipelineDescription.fragmentShader = ShaderPath("texture.frag.spv");
    pipelineDescription.vertexInput = texture_example::Vertex::GetInputDescription();
    pipelineDescription.pipelineLayout.descriptorSetLayouts.push_back(descriptor.GetLayout());
    vk_engine::GraphicsPipeline pipeline(
        engine.GetContext(), std::move(pipelineDescription), engine.GetSwapchain().GetImageFormat());
    engine.Run(
        [&](vk::CommandBuffer commandBuffer, const vk_engine::RenderTarget& target)
        {
            pipeline.EnsureCompatible(target.colorFormat);
            vk::RenderingAttachmentInfo colorAttachment{};
            colorAttachment.setImageView(target.imageView)
                .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setClearValue(vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.03F, 0.03F, 0.03F, 1.0F}}});
            vk::RenderingInfo renderingInfo{};
            renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, target.extent})
                .setLayerCount(1)
                .setColorAttachments(colorAttachment);
            commandBuffer.beginRendering(renderingInfo);
            pipeline.Bind(commandBuffer);
            commandBuffer.setViewport(0,
                                      vk::Viewport{0.0F,
                                                   0.0F,
                                                   static_cast<float>(target.extent.width),
                                                   static_cast<float>(target.extent.height),
                                                   0.0F,
                                                   1.0F});
            commandBuffer.setScissor(0, vk::Rect2D{{0, 0}, target.extent});
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics, pipeline.GetLayout(), 0, descriptor.GetSet(), {});
            commandBuffer.bindVertexBuffers(0, vertexBuffer.GetHandle(), vk::DeviceSize{0});
            commandBuffer.bindIndexBuffer(indexBuffer.GetHandle(), 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(6, 1, 0, 0, 0);
            commandBuffer.endRendering();
        });
}
