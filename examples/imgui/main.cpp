#include "vk_engine.h"
#include "vk_imgui.h"

#include <array>

#include "imgui.h"

int main()
{
    vk_engine::VkEngine engine{};
    vk_engine::VkImGui imgui(engine.GetContext(),
                             &engine.GetWindowHandle(),
                             engine.GetDrawImageFormat(),
                             2,
                             engine.GetSwapchain().GetImageCount());

    float clearColor[3] = {0.1F, 0.2F, 0.3F};
    bool showDemoWindow = true;

    engine.Run(
        [&](vk::CommandBuffer commandBuffer, vk_engine::RenderHelper& helper)
        {
            imgui.BeginFrame();
            ImGui::Begin("vk-engine ImGui Demo");
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Text("Frame time: %.3f ms", 1000.0F / ImGui::GetIO().Framerate);
            ImGui::Separator();
            ImGui::SliderFloat3("Background color", clearColor, 0.0F, 1.0F);
            ImGui::Checkbox("Show Demo Window", &showDemoWindow);
            ImGui::End();
            if (showDemoWindow)
            {
                ImGui::ShowDemoWindow(&showDemoWindow);
            }

            helper.TransitionToGraphics();
            const vk::Extent2D extent = helper.GetDrawExtent();
            vk::RenderingAttachmentInfo colorAttachment{};
            colorAttachment.setImageView(helper.GetDrawImageView())
                .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setClearValue(vk::ClearValue{
                    vk::ClearColorValue{std::array<float, 4>{clearColor[0], clearColor[1], clearColor[2], 1.0F}}});
            vk::RenderingInfo renderingInfo{};
            renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, extent})
                .setLayerCount(1)
                .setColorAttachments(colorAttachment);
            commandBuffer.beginRendering(renderingInfo);
            commandBuffer.setViewport(
                0,
                vk::Viewport{
                    0.0F, 0.0F, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0F, 1.0F});
            commandBuffer.setScissor(0, vk::Rect2D{{0, 0}, extent});
            imgui.Render(commandBuffer);
            commandBuffer.endRendering();
        });
    return 0;
}
