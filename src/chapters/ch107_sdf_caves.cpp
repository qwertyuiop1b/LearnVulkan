/**
 * @file ch107_sdf_caves.cpp
 * @brief Real-time 3D SDF cave ray marching with procedural destruction.
 */

#include <vulkan_tutorial/engine/demo_app.hpp>

#include <imgui.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace vulkan_tutorial;

namespace {
struct alignas(16) CavePush {
    glm::vec4 cameraPosition;
    glm::vec4 cameraForward;
    glm::vec4 cameraRight;
    glm::vec4 cameraUp;
    glm::vec4 viewport;    // aspect, tan(fov / 2), time, surface detail
    glm::vec4 destruction; // world-space center, radius
    glm::vec4 effects;     // blast strength, pulse phase, fog density, crystal glow
};

static_assert(sizeof(CavePush) == 112);
} // namespace

class Ch107SdfCavesApp final : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.012f, 0.016f, 0.024f};
        interactive_.camera().setTarget({0.0f, -0.35f, -1.0f});
        interactive_.camera().setDistance(5.8f);
        interactive_.camera().setAngles(145.0f, -3.0f);
        createPipeline();
    }

    void onUpdate() override {
        elapsed_ += 0.016f;
        if (autoCamera_) {
            const float yaw = 145.0f + std::sin(elapsed_ * 0.19f) * 24.0f;
            const float pitch = -2.0f + std::sin(elapsed_ * 0.27f) * 5.5f;
            interactive_.camera().setAngles(yaw, pitch);
        }
    }

    void buildUi() override {
        interactive_.buildDebugPanel("Chapter 107: GPU SDF Caves");

        ImGui::SetNextWindowPos(ImVec2(12.0f, 315.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(330.0f, 0.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("SDF Cave Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Checkbox("Automatic camera", &autoCamera_);
            ImGui::Checkbox("Animate destruction", &animateDestruction_);
            ImGui::SliderFloat3("Blast center", &destructionCenter_.x, -6.0f, 6.0f, "%.2f");
            ImGui::SliderFloat("Blast radius", &destructionRadius_, 0.0f, 4.5f, "%.2f m");
            ImGui::SliderFloat("Fracture strength", &blastStrength_, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Surface detail", &surfaceDetail_, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Fog density", &fogDensity_, 0.005f, 0.075f, "%.3f");
            ImGui::SliderFloat("Crystal glow", &crystalGlow_, 0.0f, 2.5f, "%.2f");
            if (ImGui::Button("Detonate")) {
                detonationTime_ = elapsed_;
                animateDestruction_ = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset camera")) {
                interactive_.camera().setTarget({0.0f, -0.35f, -1.0f});
                interactive_.camera().setDistance(5.8f);
                interactive_.camera().setAngles(145.0f, -3.0f);
            }
            ImGui::Separator();
            ImGui::Text("GPU: full-screen SDF ray march");
            ImGui::Text("Geometry: cave CSG + procedural formations");
            ImGui::Text("Lighting: soft shadow / AO / emissive fog");
        }
        ImGui::End();
    }

    void onRecordRender(VkCommandBuffer cmd, uint32_t) override {
        const VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent_.width),
                                  static_cast<float>(extent_.height), 0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, extent_};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        const glm::vec3 eye = interactive_.camera().eyePosition();
        const glm::vec3 forward = glm::normalize(interactive_.camera().target() - eye);
        glm::vec3 right = glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f));
        if (glm::dot(right, right) < 1.0e-5f) {
            right = glm::vec3(1.0f, 0.0f, 0.0f);
        } else {
            right = glm::normalize(right);
        }
        const glm::vec3 up = glm::normalize(glm::cross(right, forward));

        const float aspect = static_cast<float>(extent_.width) /
                             static_cast<float>(std::max(extent_.height, 1u));
        const float pulseAge = std::max(elapsed_ - detonationTime_, 0.0f);
        float animatedRadius = destructionRadius_;
        if (animateDestruction_) {
            const float expansion = 1.0f - std::exp(-pulseAge * 2.6f);
            animatedRadius *= expansion;
        }

        CavePush push{};
        push.cameraPosition = glm::vec4(eye, 1.0f);
        push.cameraForward = glm::vec4(forward, 0.0f);
        push.cameraRight = glm::vec4(right, 0.0f);
        push.cameraUp = glm::vec4(up, 0.0f);
        push.viewport = glm::vec4(aspect, std::tan(glm::radians(52.0f) * 0.5f), elapsed_, surfaceDetail_);
        push.destruction = glm::vec4(destructionCenter_, animatedRadius);
        push.effects = glm::vec4(blastStrength_, pulseAge, fogDensity_, crystalGlow_);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    void onShutdown() override {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    }

  private:
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    float elapsed_ = 0.0f;
    float detonationTime_ = -4.0f;
    glm::vec3 destructionCenter_{1.2f, -0.3f, -3.8f};
    float destructionRadius_ = 2.7f;
    float blastStrength_ = 0.75f;
    float surfaceDetail_ = 0.78f;
    float fogDensity_ = 0.032f;
    float crystalGlow_ = 1.25f;
    bool autoCamera_ = true;
    bool animateDestruction_ = true;

    void createPipeline() {
        const VkPushConstantRange pushRange{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(CavePush)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_));

        const VkShaderModule vertexShader = createShaderModuleFromFile(device_, "sdf_cave_fullscreen.vert.spv");
        const VkShaderModule fragmentShader = createShaderModuleFromFile(device_, "sdf_cave_raymarch.frag.spv");
        const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
             vertexShader, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
             fragmentShader, "main"},
        }};

        const VkPipelineVertexInputStateCreateInfo vertexInput{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterization{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blending.attachmentCount = 1;
        blending.pAttachments = &blendAttachment;

        const std::array<VkDynamicState, 2> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT,
                                                          VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();

        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterization;
        pipelineInfo.pMultisampleState = &multisample;
        pipelineInfo.pColorBlendState = &blending;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = pipelineLayout_;
        pipelineInfo.renderPass = renderPass_;
        pipelineInfo.subpass = 0;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_));

        vkDestroyShaderModule(device_, fragmentShader, nullptr);
        vkDestroyShaderModule(device_, vertexShader, nullptr);
    }
};

int main() {
    try {
        Ch107SdfCavesApp app;
        app.run("ch107 - GPU SDF Caves and Destruction", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch107 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
