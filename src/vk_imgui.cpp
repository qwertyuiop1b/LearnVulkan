#include "vk_imgui.h"

#include <stdexcept>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

namespace vk_engine
{

VkImGui::VkImGui(const VkContext& inContext,
                 GLFWwindow* inWindow,
                 vk::Format drawImageFormat,
                 uint32_t minImageCount,
                 uint32_t imageCount)
{
    IMGUI_CHECKVERSION();
    imguiContext = ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplGlfw_InitForVulkan(inWindow, true);

    const VkFormat colorFormat = static_cast<VkFormat>(drawImageFormat);
    VkPipelineRenderingCreateInfo pipelineRenderingInfo{};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    pipelineRenderingInfo.colorAttachmentCount = 1;
    pipelineRenderingInfo.pColorAttachmentFormats = &colorFormat;

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = *inContext.GetInstance();
    initInfo.PhysicalDevice = *inContext.GetPhysicalDevice();
    initInfo.Device = *inContext.GetDevice();
    initInfo.QueueFamily = inContext.GetGraphicQueueFamilyIndex();
    initInfo.Queue = *inContext.GetGraphicQueue();
    initInfo.DescriptorPoolSize = 1024;
    initInfo.MinImageCount = minImageCount;
    initInfo.ImageCount = imageCount;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = pipelineRenderingInfo;

    if (!ImGui_ImplVulkan_Init(&initInfo))
    {
        throw std::runtime_error("failed to initialize ImGui Vulkan backend");
    }
}

VkImGui::~VkImGui()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(imguiContext);
}

void VkImGui::BeginFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void VkImGui::Render(vk::CommandBuffer commandBuffer)
{
    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    if (drawData != nullptr)
    {
        ImGui_ImplVulkan_RenderDrawData(drawData, static_cast<VkCommandBuffer>(commandBuffer));
    }
}
} // namespace vk_engine
