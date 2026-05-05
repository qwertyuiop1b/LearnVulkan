#pragma once

#include "Qdevice.hpp"
#include "vulkan/vulkan_core.h"
#include <cstdint>
#include <string>
#include <vector>

namespace q_vulkan {

struct PipelineConfigInfo {
    VkViewport viewport;
    VkRect2D scissor; 
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
    VkPipelineRasterizationStateCreateInfo rasterizationInfo;
    VkPipelineMultisampleStateCreateInfo multisampleInfo;
    VkPipelineColorBlendAttachmentState colorBlendAttachment;
    VkPipelineColorBlendStateCreateInfo colorBlendInfo;
    VkPipelineDepthStencilStateCreateInfo depthStencilInfo;
    VkPipelineLayout pipelineLayout = nullptr;
    VkRenderPass renderPass = nullptr;
    uint32_t subpass = 0;
};

class QPipeline {
public:
    QPipeline(QDevice& device, const std::string& vertPath, const std::string& fragPath, const PipelineConfigInfo& pipelineInfo);

    ~QPipeline();

    QPipeline(const QPipeline&) = delete;

    QPipeline& operator=(const QPipeline&) = delete;

    void bind(VkCommandBuffer commandBuffer);

    static PipelineConfigInfo defaultPipelineConfigInfo(uint32_t width, uint32_t height);

private:
    const std::string vertPath;
    const std::string fragPath;

    static std::vector<char> readFile(const std::string& filepath);

    void createGraphicPipeline(const PipelineConfigInfo& pipelineInfo);

    void createShaderModule(const std::vector<char>& code, VkShaderModule* shaderModule);

    QDevice& device;
    VkPipeline graphicsPipeline;
    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;
};
};