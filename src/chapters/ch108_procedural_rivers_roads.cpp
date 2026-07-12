/**
 * @file ch108_procedural_rivers_roads.cpp
 * @brief Procedural terrain with GPU-generated river and road ribbons.
 */

#include <vulkan_tutorial/engine/demo_app.hpp>

#include <imgui.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <iostream>

namespace {
struct alignas(16) WorldPush {
    glm::mat4 viewProjection;
    glm::vec4 cameraPosition;
    glm::vec4 params;
    glm::vec4 controls;
};
static_assert(sizeof(WorldPush) == 112);
}

class Ch108App final : public DemoApp {
  protected:
    bool needsDepthBuffer() const override { return true; }

    void onInit() override {
        interactive_.camera().setTarget({0.0f, 0.0f, 0.0f});
        interactive_.camera().setDistance(42.0f);
        interactive_.camera().setAngles(32.0f, 31.0f);
        createPipeline();
    }

    void onUpdate() override { elapsed_ += 0.016f; }

    void buildUi() override {
        interactive_.buildDebugPanel("第108章：程序化河流与道路");
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("World Generator", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderFloat("Terrain Size", &terrainScale_, 45.0f, 120.0f, "%.0f m");
            ImGui::SliderFloat("River Width", &riverWidth_, 2.0f, 10.0f, "%.1f m");
            ImGui::SliderFloat("Road Width", &roadWidth_, 1.0f, 6.0f, "%.1f m");
            ImGui::Text("GPU terrain: 112x112 cells");
            ImGui::Text("River/Road: 192 segments each");
        }
        ImGui::End();
    }

    void onRecordRender(VkCommandBuffer cmd, uint32_t) override {
        VkViewport viewport{0, 0, float(extent_.width), float(extent_.height), 0, 1};
        VkRect2D scissor{{0, 0}, extent_};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        const float aspect = float(extent_.width) / float(std::max(1u, extent_.height));
        WorldPush push{};
        push.viewProjection = interactive_.camera().projectionMatrix(aspect, 55.0f, 0.1f, 800.0f) *
                              interactive_.camera().viewMatrix();
        push.cameraPosition = glm::vec4(interactive_.camera().eyePosition(), 1.0f);
        push.controls = {riverWidth_, roadWidth_, 3.6f, 0.0f};

        push.params = {elapsed_, 0.0f, 0.0f, terrainScale_};
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(cmd, (112 - 1) * (112 - 1) * 6, 1, 0, 0);

        push.params.y = 1.0f;
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(cmd, 192 * 6, 1, 0, 0);

        push.params.y = 2.0f;
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(cmd, 192 * 6, 1, 0, 0);
    }

    void onShutdown() override {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, layout_, nullptr);
    }

  private:
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    float elapsed_ = 0.0f;
    float terrainScale_ = 78.0f;
    float riverWidth_ = 5.8f;
    float roadWidth_ = 3.0f;

    void createPipeline() {
        VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(WorldPush)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &layout_));

        VkShaderModule vert = createShaderModuleFromFile(device_, "procedural_world.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "procedural_world.frag.spv");
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
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState colorAttachment{};
        colorAttachment.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &colorAttachment;
        std::array<VkDynamicState, 2> states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<uint32_t>(states.size());
        dynamic.pDynamicStates = states.data();

        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.stageCount = static_cast<uint32_t>(stages.size());
        info.pStages = stages.data();
        info.pVertexInputState = &vertexInput;
        info.pInputAssemblyState = &assembly;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depth;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic;
        info.layout = layout_;
        info.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }
};

int main() {
    try {
        Ch108App app;
        app.run("ch108 - Procedural Rivers and Roads", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch108 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
