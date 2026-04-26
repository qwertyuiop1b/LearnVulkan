/**
 * @file ch36_gpl.cpp
 * @brief 第36章：Graphics Pipeline Library（VK_EXT_graphics_pipeline_library）
 *
 * 将完整图形管线拆成多个「管线库」片段（顶点输入 / 光栅化前着色器 / 片段着色器 /
 * 片段输出接口），再链接成可执行管线。可减少 PSO 编译卡顿、复用片段。
 *
 * 依赖：VK_KHR_pipeline_library + VK_EXT_graphics_pipeline_library。
 * 若当前设备不支持（部分平台/MoltenVK 可能无此扩展），自动回退为传统单体管线。
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

constexpr uint32_t WIDTH                  = 800;
constexpr uint32_t HEIGHT                 = 600;
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;

class Ch36App {
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow*      window_         = nullptr;
    VkInstance       instance_       = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_        = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          graphicsQueue_  = VK_NULL_HANDLE;
    VkQueue          presentQueue_   = VK_NULL_HANDLE;
    VkSwapchainKHR   swapchain_      = VK_NULL_HANDLE;
    VkRenderPass     renderPass_     = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline       pipeline_       = VK_NULL_HANDLE;
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;

    std::vector<VkPipeline>      gplLibraries_;
    bool                         useGpl_ = false;

    std::vector<VkImage>         swapchainImages_;
    std::vector<VkImageView>     swapchainImageViews_;
    std::vector<VkFramebuffer>   framebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkFormat                     swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D                   swapchainExtent_{};
    QueueFamilyIndices           queueIndices_;

    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence>     inFlightFences_;
    uint32_t                 currentFrame_ = 0;
    bool                     framebufferResized_ = false;

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT,
            "Ch36 - Graphics Pipeline Library（GPL）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* win, int /*w*/, int /*h*/)
    {
        reinterpret_cast<Ch36App*>(glfwGetWindowUserPointer(win))->framebufferResized_ = true;
    }

    void initVulkan()
    {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
        if (useGpl_)
            std::cout << "\n✅ 使用 GPL：4 个管线库 + 链接管线\n";
        else
            std::cout << "\n✅ 使用单体图形管线（设备未暴露 GPL 或特性未开启）\n";
    }

    static bool hasDeviceExtension(VkPhysicalDevice dev, const char* name)
    {
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> exts(n);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, exts.data());
        for (const auto& e : exts) {
            if (std::strcmp(e.extensionName, name) == 0)
                return true;
        }
        return false;
    }

    static bool queryGplFeatureSupport(VkPhysicalDevice dev)
    {
        VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl{};
        gpl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &gpl;
        vkGetPhysicalDeviceFeatures2(dev, &f2);
        return gpl.graphicsPipelineLibrary == VK_TRUE;
    }

    void pickPhysicalDevice()
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        for (auto& d : devices) {
            if (!findQueueFamilies(d, surface_).isComplete())
                continue;
            if (!checkDeviceExtensionSupport(d))
                continue;
            SwapChainSupportDetails sc = querySwapChainSupport(d, surface_);
            if (sc.formats.empty() || sc.presentModes.empty())
                continue;
            physicalDevice_ = d;
            const bool extPl  = hasDeviceExtension(d, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
            const bool extGpl = hasDeviceExtension(d, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
            useGpl_           = extPl && extGpl && queryGplFeatureSupport(d);
            break;
        }
        if (physicalDevice_ == VK_NULL_HANDLE)
            throw std::runtime_error("无合适 GPU");
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(physicalDevice_, &p);
        std::cout << "✅ GPU: " << p.deviceName << "\n";
        if (useGpl_)
            std::cout << "   VK_EXT_graphics_pipeline_library：可用\n";
        else
            std::cout << "   VK_EXT_graphics_pipeline_library：不可用 → 回退单体管线\n";
    }

    void createLogicalDevice()
    {
        queueIndices_ = findQueueFamilies(physicalDevice_, surface_);
        std::set<uint32_t> fams = { queueIndices_.graphicsFamily.value(),
            queueIndices_.presentFamily.value() };
        const float                    pri  = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t f : fams) {
            VkDeviceQueueCreateInfo q{};
            q.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueFamilyIndex = f;
            q.queueCount       = 1;
            q.pQueuePriorities = &pri;
            qcis.push_back(q);
        }

        std::vector<const char*> devExts = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            "VK_KHR_portability_subset",
        };
        if (useGpl_) {
            devExts.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
            devExts.push_back(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
        }

        VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gplFeat{};
        gplFeat.sType                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT;
        gplFeat.graphicsPipelineLibrary = useGpl_ ? VK_TRUE : VK_FALSE;

        VkPhysicalDeviceFeatures2 feat2{};
        feat2.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feat2.features.samplerAnisotropy = VK_TRUE;
        feat2.pNext                 = useGpl_ ? &gplFeat : nullptr;

        VkDeviceCreateInfo ci{};
        ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.pNext                   = useGpl_ ? &feat2 : nullptr;
        ci.queueCreateInfoCount    = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos       = qcis.data();
        ci.pEnabledFeatures        = useGpl_ ? nullptr : &feat2.features;
        ci.enabledExtensionCount   = static_cast<uint32_t>(devExts.size());
        ci.ppEnabledExtensionNames = devExts.data();
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount   = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_));
        vkGetDeviceQueue(device_, queueIndices_.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueIndices_.presentFamily.value(), 0, &presentQueue_);
    }

    void createSwapchain()
    {
        SwapChainSupportDetails sc  = querySwapChainSupport(physicalDevice_, surface_);
        VkSurfaceFormatKHR      fmt = chooseSwapSurfaceFormat(sc.formats);
        VkPresentModeKHR        mode = chooseSwapPresentMode(sc.presentModes);
        swapchainExtent_ = chooseSwapExtent(sc.capabilities, window_);
        uint32_t n = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            n = std::min(n, sc.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{};
        ci.sType                 = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface               = surface_;
        ci.minImageCount         = n;
        ci.imageFormat           = fmt.format;
        ci.imageColorSpace       = fmt.colorSpace;
        ci.imageExtent           = swapchainExtent_;
        ci.imageArrayLayers      = 1;
        ci.imageUsage            = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform          = sc.capabilities.currentTransform;
        ci.compositeAlpha        = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode           = mode;
        ci.clipped               = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
        swapchainImages_.resize(n);
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, swapchainImages_.data());
        swapchainImageFormat_ = fmt.format;
    }

    void createImageViews()
    {
        swapchainImageViews_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image    = swapchainImages_[i];
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format   = swapchainImageFormat_;
            ci.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
            ci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]));
        }
    }

    void createRenderPass()
    {
        VkAttachmentDescription ca{};
        ca.format         = swapchainImageFormat_;
        ca.samples        = VK_SAMPLE_COUNT_1_BIT;
        ca.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        ca.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ca.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ca.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        ca.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference cr{};
        cr.attachment = 0;
        cr.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkSubpassDescription sp{};
        sp.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments    = &cr;
        VkSubpassDependency dep{};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass    = 0;
        dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo rpi{};
        rpi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = 1;
        rpi.pAttachments    = &ca;
        rpi.subpassCount    = 1;
        rpi.pSubpasses      = &sp;
        rpi.dependencyCount = 1;
        rpi.pDependencies   = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_));
    }

    void createGraphicsPipeline()
    {
        VkShaderModule vert = createShaderModuleFromFile(device_, "triangle.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "triangle.frag.spv");

        VkPipelineShaderStageCreateInfo vertStage{};
        vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        vertStage.module = vert;
        vertStage.pName  = "main";
        VkPipelineShaderStageCreateInfo fragStage{};
        fragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStage.module = frag;
        fragStage.pName  = "main";

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vs{};
        vs.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vs.viewportCount = 1;
        vs.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.lineWidth   = 1.0f;
        rs.cullMode    = VK_CULL_MODE_BACK_BIT;
        rs.frontFace   = VK_FRONT_FACE_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments    = &cba;

        std::vector<VkDynamicState> dyn = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo ds{};
        ds.sType           = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        ds.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        ds.pDynamicStates    = dyn.data();

        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &pipelineLayout_));

        if (!useGpl_) {
            VkGraphicsPipelineCreateInfo pi{};
            pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pi.stageCount          = 2;
            VkPipelineShaderStageCreateInfo stages[2] = { vertStage, fragStage };
            pi.pStages             = stages;
            pi.pVertexInputState   = &vi;
            pi.pInputAssemblyState = &ia;
            pi.pViewportState      = &vs;
            pi.pRasterizationState = &rs;
            pi.pMultisampleState   = &ms;
            pi.pColorBlendState    = &cb;
            pi.pDynamicState       = &ds;
            pi.layout              = pipelineLayout_;
            pi.renderPass          = renderPass_;
            VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
            vkDestroyShaderModule(device_, frag, nullptr);
            vkDestroyShaderModule(device_, vert, nullptr);
            return;
        }

        VkGraphicsPipelineLibraryCreateInfoEXT libInfoVi{};
        libInfoVi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT;
        libInfoVi.flags = VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT;
        VkGraphicsPipelineCreateInfo viLib{};
        viLib.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        viLib.pNext               = &libInfoVi;
        viLib.flags               = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
        viLib.pVertexInputState   = &vi;
        viLib.pInputAssemblyState = &ia;
        viLib.layout              = pipelineLayout_;

        VkGraphicsPipelineLibraryCreateInfoEXT libInfoPr{};
        libInfoPr.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT;
        libInfoPr.flags = VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT;
        VkGraphicsPipelineCreateInfo prLib{};
        prLib.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        prLib.pNext               = &libInfoPr;
        prLib.flags               = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
        prLib.stageCount          = 1;
        prLib.pStages             = &vertStage;
        prLib.pViewportState      = &vs;
        prLib.pRasterizationState = &rs;
        prLib.pDynamicState       = &ds;
        prLib.layout              = pipelineLayout_;
        prLib.renderPass          = renderPass_;
        prLib.subpass             = 0;

        VkGraphicsPipelineLibraryCreateInfoEXT libInfoFs{};
        libInfoFs.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT;
        libInfoFs.flags = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT;
        VkGraphicsPipelineCreateInfo fsLib{};
        fsLib.sType     = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        fsLib.pNext     = &libInfoFs;
        fsLib.flags     = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
        fsLib.stageCount = 1;
        fsLib.pStages   = &fragStage;
        fsLib.layout    = pipelineLayout_;
        fsLib.renderPass = renderPass_;
        fsLib.subpass   = 0;

        VkGraphicsPipelineLibraryCreateInfoEXT libInfoFo{};
        libInfoFo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT;
        libInfoFo.flags = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT;
        VkGraphicsPipelineCreateInfo foLib{};
        foLib.sType             = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        foLib.pNext             = &libInfoFo;
        foLib.flags             = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
        foLib.pMultisampleState = &ms;
        foLib.pColorBlendState  = &cb;
        foLib.layout            = pipelineLayout_;
        foLib.renderPass        = renderPass_;
        foLib.subpass           = 0;

        gplLibraries_.resize(4);
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &viLib, nullptr, &gplLibraries_[0]));
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &prLib, nullptr, &gplLibraries_[1]));
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &fsLib, nullptr, &gplLibraries_[2]));
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &foLib, nullptr, &gplLibraries_[3]));

        VkPipelineLibraryCreateInfoKHR linkInfo{};
        linkInfo.sType        = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR;
        linkInfo.libraryCount = static_cast<uint32_t>(gplLibraries_.size());
        linkInfo.pLibraries   = gplLibraries_.data();

        VkGraphicsPipelineCreateInfo linked{};
        linked.sType      = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        linked.pNext      = &linkInfo;
        linked.stageCount = 0;
        linked.layout     = pipelineLayout_;
        linked.renderPass = renderPass_;
        linked.subpass    = 0;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &linked, nullptr, &pipeline_));

        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }

    void createFramebuffers()
    {
        framebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            VkImageView           att = swapchainImageViews_[i];
            VkFramebufferCreateInfo ci{};
            ci.sType       = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass  = renderPass_;
            ci.attachmentCount = 1;
            ci.pAttachments    = &att;
            ci.width       = swapchainExtent_.width;
            ci.height      = swapchainExtent_.height;
            ci.layers      = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
        }
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{};
        ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
    }

    void createCommandBuffers()
    {
        commandBuffers_.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = commandPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()));
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences_.resize(MAX_FRAMES_IN_FLIGHT);
        VkSemaphoreCreateInfo semCI{};
        semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &semCI, nullptr, &imageAvailableSemaphores_[i]));
            VK_CHECK(vkCreateSemaphore(device_, &semCI, nullptr, &renderFinishedSemaphores_[i]));
            VK_CHECK(vkCreateFence(device_, &fenceCI, nullptr, &inFlightFences_[i]));
        }
    }

    void drawFrame()
    {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex = 0;
        VkResult result = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
            imageAvailableSemaphores_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("获取交换链图像失败");
        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordCommandBuffer(commandBuffers_[currentFrame_], imageIndex);
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        VkSemaphore          waitSem[] = { imageAvailableSemaphores_[currentFrame_] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount   = 1;
        submitInfo.pWaitSemaphores      = waitSem;
        submitInfo.pWaitDstStageMask    = waitStages;
        submitInfo.commandBufferCount   = 1;
        submitInfo.pCommandBuffers      = &commandBuffers_[currentFrame_];
        VkSemaphore signalSem[]         = { renderFinishedSemaphores_[currentFrame_] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores    = signalSem;
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &submitInfo, inFlightFences_[currentFrame_]));
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores    = signalSem;
        VkSwapchainKHR swapchains[] = { swapchain_ };
        presentInfo.swapchainCount  = 1;
        presentInfo.pSwapchains     = swapchains;
        presentInfo.pImageIndices   = &imageIndex;
        result = vkQueuePresentKHR(presentQueue_, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized_) {
            framebufferResized_ = false;
            recreateSwapchain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("呈现图像失败");
        }
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void recreateSwapchain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window_, &width, &height);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        cleanupSwapchain();
        createSwapchain();
        createImageViews();
        createFramebuffers();
    }

    void cleanupSwapchain()
    {
        for (auto fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        VkClearValue clearColor = {{{ 0.02f, 0.02f, 0.05f, 1.0f }}};
        VkRenderPassBeginInfo rpBI{};
        rpBI.sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBI.renderPass  = renderPass_;
        rpBI.framebuffer = framebuffers_[imageIndex];
        rpBI.renderArea  = { { 0, 0 }, swapchainExtent_ };
        rpBI.clearValueCount = 1;
        rpBI.pClearValues    = &clearColor;
        vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkViewport vp{ 0.0f, 0.0f, static_cast<float>(swapchainExtent_.width),
            static_cast<float>(swapchainExtent_.height), 0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ { 0, 0 }, swapchainExtent_ };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void mainLoop()
    {
        std::cout << "\n🎨 彩色三角形。ESC 退出。\n";
        uint64_t frameCount = 0;
        auto     startTime  = std::chrono::steady_clock::now();
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
            ++frameCount;
            auto now     = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<float>(now - startTime).count();
            if (elapsed >= 5.0f) {
                std::cout << "📊 FPS: " << static_cast<uint64_t>(frameCount / elapsed) << "\n";
                frameCount = 0;
                startTime  = now;
            }
        }
        vkDeviceWaitIdle(device_);
    }

    void cleanup()
    {
        cleanupSwapchain();
        vkDestroyPipeline(device_, pipeline_, nullptr);
        for (VkPipeline lib : gplLibraries_)
            vkDestroyPipeline(device_, lib, nullptr);
        gplLibraries_.clear();
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSemaphores_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
        std::cout << "✅ 清理完成。\n";
    }

    void createInstance()
    {
        VkApplicationInfo ai{};
        ai.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.apiVersion = VK_API_VERSION_1_3;
        auto exts     = getRequiredInstanceExtensions();
        VkInstanceCreateInfo ci{};
        ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo        = &ai;
        ci.enabledExtensionCount   = static_cast<uint32_t>(exts.size());
        ci.ppEnabledExtensionNames = exts.data();
        ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount   = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    }

    void createSurface()
    {
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
    }
};

int main()
{
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << " 第36章：Graphics Pipeline Library（GPL）\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    Ch36App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
