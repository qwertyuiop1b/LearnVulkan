/**
 * @file ch28_subgroup.cpp
 * @brief 第28章：Subgroup Operations（Wave Intrinsics）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【什么是 Subgroup？】
 *
 *  GPU 以 SIMD（Single Instruction Multiple Data）方式执行着色器：
 *  同一时刻，一批线程（称为 Subgroup/Wave/Warp）执行相同指令。
 *
 *  NVIDIA：Warp = 32 线程
 *  AMD：   Wave = 64 线程（RDNA2 也支持 Wave32）
 *  ARM:    Subgroup = 4-16 线程
 *
 * 【为什么 Subgroup 操作快？】
 *
 *  普通共享内存规约（prefix sum，N 个元素）：
 *    O(log N) 轮，每轮需要 barrier + shared memory 读写
 *
 *  Subgroup 规约：
 *    单条指令完成 subgroup 内所有线程的规约
 *    无需 barrier，无需共享内存！硬件直接完成通信
 *    比共享内存快 3-10 倍
 *
 * 【常用 Subgroup 扩展】
 *
 *  GL_KHR_shader_subgroup_basic       → subgroupElect, gl_SubgroupSize 等
 *  GL_KHR_shader_subgroup_arithmetic  → subgroupAdd, subgroupMin/Max 等
 *  GL_KHR_shader_subgroup_ballot      → subgroupBallot (收集所有线程的 bool)
 *  GL_KHR_shader_subgroup_shuffle     → subgroupShuffle (线程间数据交换)
 *  GL_KHR_shader_subgroup_vote        → subgroupAll/Any (投票)
 *
 * 【本章示例】
 *
 *  1. 大数组并行求和（对比 CPU 串行 vs GPU Subgroup 方案）
 *  2. 展示 subgroupAdd 的两层规约模式
 *  3. 打印 subgroup 大小等 GPU 信息
 *
 * 【典型使用场景】
 *
 *  - 快速规约（sum/min/max/and/or）：粒子统计、遮挡剔除投票
 *  - 前缀求和（prefix scan）：直方图、压缩算法
 *  - 线程投票（ballot/vote）：条件分支优化
 *  - 纹理坐标插值：相邻线程的 dFdx/dFdy 差分
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

constexpr uint32_t N_ELEMENTS = 1024 * 1024;   // 100万个 float
constexpr uint32_t WG_SIZE    = 256;            // local_size_x
constexpr uint32_t N_WORKGROUPS = (N_ELEMENTS + WG_SIZE - 1) / WG_SIZE;

struct SubgroupResult {
    float    partialSums[N_WORKGROUPS];  // 每个 workgroup 的部分和
    uint32_t subgroupSize;
    uint32_t subgroupCount;
};

class Ch28App {
public:
    void run()
    {
        init();
        benchmark();
        cleanup();
    }

private:
    VkInstance       instance_       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          computeQueue_   = VK_NULL_HANDLE;
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;

    VkBuffer       inputBuffer_    = VK_NULL_HANDLE;
    VkDeviceMemory inputMemory_    = VK_NULL_HANDLE;
    VkBuffer       outputBuffer_   = VK_NULL_HANDLE;
    VkDeviceMemory outputMemory_   = VK_NULL_HANDLE;
    void*          outputMapped_   = nullptr;

    VkDescriptorSetLayout setLayout_    = VK_NULL_HANDLE;
    VkPipelineLayout      pipeLayout_   = VK_NULL_HANDLE;
    VkPipeline            pipeline_     = VK_NULL_HANDLE;
    VkDescriptorPool      descPool_     = VK_NULL_HANDLE;
    VkDescriptorSet       descSet_      = VK_NULL_HANDLE;

    uint32_t computeQueueFamily_ = 0;
    float    subgroupSizeGPU_    = 0;

    void init()
    {
        createInstance();
        pickPhysicalDevice();
        createLogicalDeviceWithSubgroupFeatures();
        createCommandPool();
        createBuffers();
        createDescriptorSetLayout();
        createComputePipeline();
        createDescriptorPool();
        createDescriptorSet();
    }

    void benchmark()
    {
        // ── 生成随机数据 ──────────────────────────────────────────────────
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> inputData(N_ELEMENTS);
        for (auto& v : inputData) v = dist(rng);

        // 上传到 GPU
        VkDeviceSize inputSz = sizeof(float) * N_ELEMENTS;
        void* mapped = nullptr;
        vkMapMemory(device_, inputMemory_, 0, inputSz, 0, &mapped);
        std::memcpy(mapped, inputData.data(), inputSz);
        vkUnmapMemory(device_, inputMemory_);

        // ── CPU 基准（单线程求和）───────────────────────────────────────
        auto cpuStart = std::chrono::high_resolution_clock::now();
        double cpuSum = 0.0;
        for (float v : inputData) cpuSum += v;
        double cpuMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuStart).count();

        // ── GPU Subgroup 规约 ────────────────────────────────────────────
        // 步骤1：dispatch compute shader
        auto gpuStart = std::chrono::high_resolution_clock::now();

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device_, &ai, &cmd);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeLayout_, 0, 1, &descSet_, 0, nullptr);
        vkCmdDispatch(cmd, N_WORKGROUPS, 1, 1);

        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;si.commandBufferCount=1;si.pCommandBuffers=&cmd;
        vkQueueSubmit(computeQueue_, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(computeQueue_);
        vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);

        // 步骤2：在 CPU 上做最终规约（将 N_WORKGROUPS 个部分和相加）
        SubgroupResult* result = reinterpret_cast<SubgroupResult*>(outputMapped_);
        double gpuPartialMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - gpuStart).count();

        double gpuSum = 0.0;
        for (uint32_t i = 0; i < N_WORKGROUPS; ++i) gpuSum += result->partialSums[i];

        subgroupSizeGPU_ = static_cast<float>(result->subgroupSize);
        uint32_t subgroupCount = result->subgroupCount;

        // ── 打印结果 ──────────────────────────────────────────────────────
        std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
        std::cout << "║  第28章：Subgroup Operations 性能测试                      ║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
        std::cout << "║  数据集：" << N_ELEMENTS / 1024 / 1024 << "M 个 float\n";
        std::cout << "║  GPU: subgroupSize  = " << result->subgroupSize << " 线程\n";
        std::cout << "║  GPU: subgroupCount = " << subgroupCount << " 个 subgroup/workgroup\n";
        std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
        std::cout << "║  CPU 串行求和  : " << cpuMs     << " ms → sum = " << cpuSum << "\n";
        std::cout << "║  GPU Subgroup  : " << gpuPartialMs << " ms → sum = " << gpuSum << "\n";
        std::cout << "║  精度误差      : " << std::abs(gpuSum - cpuSum) / cpuSum * 100 << "%\n";
        std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
        std::cout << "\n✅ 两级规约流程：\n";
        std::cout << "   Level 1: subgroupAdd() → 每 " << result->subgroupSize
                  << " 个线程合并为 1 个部分和（GPU 硬件指令）\n";
        std::cout << "   Level 2: 共享内存规约 → 每 " << subgroupCount
                  << " 个 subgroup 合并为 1 个 workgroup 和\n";
        std::cout << "   Level 3: CPU 将 " << N_WORKGROUPS << " 个 workgroup 和相加\n";
    }

    void createInstance()
    {
        VkApplicationInfo ai{};ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;ai.apiVersion=VK_API_VERSION_1_3;
        auto exts=getRequiredInstanceExtensions();
        VkInstanceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ci.pApplicationInfo=&ai;ci.enabledExtensionCount=static_cast<uint32_t>(exts.size());ci.ppEnabledExtensionNames=exts.data();ci.flags|=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}
        VK_CHECK(vkCreateInstance(&ci,nullptr,&instance_));
    }

    void pickPhysicalDevice()
    {
        uint32_t c=0;vkEnumeratePhysicalDevices(instance_,&c,nullptr);
        std::vector<VkPhysicalDevice> devs(c);vkEnumeratePhysicalDevices(instance_,&c,devs.data());
        physicalDevice_=devs[0];  // 取第一个

        VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(physicalDevice_,&p);
        std::cout<<"✅ GPU: "<<p.deviceName<<"\n";

        // 查询 Subgroup 属性
        VkPhysicalDeviceSubgroupProperties subgroupProps{};
        subgroupProps.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext=&subgroupProps;
        vkGetPhysicalDeviceProperties2(physicalDevice_,&props2);

        std::cout<<"🔷 Subgroup 大小：" << subgroupProps.subgroupSize << " 线程\n";
        std::cout<<"🔷 支持的 Subgroup 操作：0x"
                  << std::hex << subgroupProps.supportedOperations << std::dec << "\n";

        // 查找 Compute 队列族
        uint32_t qfCount=0;vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_,&qfCount,nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_,&qfCount,qfs.data());
        for(uint32_t i=0;i<qfCount;++i){
            if(qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){computeQueueFamily_=i;break;}
        }
    }

    void createLogicalDeviceWithSubgroupFeatures()
    {
        const float pri=1.0f;
        VkDeviceQueueCreateInfo qci{};qci.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;qci.queueFamilyIndex=computeQueueFamily_;qci.queueCount=1;qci.pQueuePriorities=&pri;

        // Subgroup 特性（Vulkan 1.1 核心，无需额外扩展）
        // 只需要在着色器中声明 GL_KHR_shader_subgroup_* 扩展即可
        VkPhysicalDeviceFeatures feat{};
        VkDeviceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;ci.queueCreateInfoCount=1;ci.pQueueCreateInfos=&qci;ci.pEnabledFeatures=&feat;
        ci.enabledExtensionCount=static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        ci.ppEnabledExtensionNames=DEVICE_EXTENSIONS.data();
        if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}
        VK_CHECK(vkCreateDevice(physicalDevice_,&ci,nullptr,&device_));
        vkGetDeviceQueue(device_,computeQueueFamily_,0,&computeQueue_);
    }

    void createCommandPool(){VkCommandPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;ci.queueFamilyIndex=computeQueueFamily_;VK_CHECK(vkCreateCommandPool(device_,&ci,nullptr,&commandPool_));}

    uint32_t findMemoryType(uint32_t f,VkMemoryPropertyFlags p){VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(physicalDevice_,&mp);for(uint32_t i=0;i<mp.memoryTypeCount;++i)if((f&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&p)==p)return i;throw std::runtime_error("找不到内存类型");}

    void createBuffer(VkDeviceSize sz,VkBufferUsageFlags u,VkMemoryPropertyFlags p,VkBuffer&b,VkDeviceMemory&m,void**mapped=nullptr)
    {
        VkBufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;ci.size=sz;ci.usage=u;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_,&ci,nullptr,&b));
        VkMemoryRequirements mr;vkGetBufferMemoryRequirements(device_,b,&mr);
        VkMemoryAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;ai.allocationSize=mr.size;ai.memoryTypeIndex=findMemoryType(mr.memoryTypeBits,p);
        VK_CHECK(vkAllocateMemory(device_,&ai,nullptr,&m));VK_CHECK(vkBindBufferMemory(device_,b,m,0));
        if(mapped)vkMapMemory(device_,m,0,sz,0,mapped);
    }

    void createBuffers()
    {
        VkDeviceSize inputSz  = sizeof(float) * N_ELEMENTS;
        VkDeviceSize outputSz = sizeof(SubgroupResult);
        createBuffer(inputSz,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            inputBuffer_, inputMemory_);
        createBuffer(outputSz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            outputBuffer_, outputMemory_, &outputMapped_);
    }

    void createDescriptorSetLayout()
    {
        std::array<VkDescriptorSetLayoutBinding,2> bindings{};
        bindings[0]={0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        bindings[1]={1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;ci.bindingCount=2;ci.pBindings=bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_,&ci,nullptr,&setLayout_));
    }

    void createComputePipeline()
    {
        VkShaderModule comp=createShaderModuleFromFile(device_,"subgroup_reduce.comp.spv");
        VkPipelineShaderStageCreateInfo stage{};stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;stage.module=comp;stage.pName="main";
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.setLayoutCount=1;pli.pSetLayouts=&setLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_,&pli,nullptr,&pipeLayout_));
        VkComputePipelineCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;ci.stage=stage;ci.layout=pipeLayout_;
        VK_CHECK(vkCreateComputePipelines(device_,VK_NULL_HANDLE,1,&ci,nullptr,&pipeline_));
        vkDestroyShaderModule(device_,comp,nullptr);
    }

    void createDescriptorPool()
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2};
        VkDescriptorPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;ci.poolSizeCount=1;ci.pPoolSizes=&ps;ci.maxSets=1;
        VK_CHECK(vkCreateDescriptorPool(device_,&ci,nullptr,&descPool_));
    }

    void createDescriptorSet()
    {
        VkDescriptorSetAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;ai.descriptorPool=descPool_;ai.descriptorSetCount=1;ai.pSetLayouts=&setLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_,&ai,&descSet_));
        VkDescriptorBufferInfo inBI{inputBuffer_,0,VK_WHOLE_SIZE};
        VkDescriptorBufferInfo outBI{outputBuffer_,0,VK_WHOLE_SIZE};
        std::array<VkWriteDescriptorSet,2> ws{};
        ws[0]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,descSet_,0,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&inBI,nullptr};
        ws[1]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,descSet_,1,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&outBI,nullptr};
        vkUpdateDescriptorSets(device_,2,ws.data(),0,nullptr);
    }

    void cleanup()
    {
        vkUnmapMemory(device_,outputMemory_);
        vkDestroyBuffer(device_,inputBuffer_,nullptr);vkFreeMemory(device_,inputMemory_,nullptr);
        vkDestroyBuffer(device_,outputBuffer_,nullptr);vkFreeMemory(device_,outputMemory_,nullptr);
        vkDestroyDescriptorPool(device_,descPool_,nullptr);
        vkDestroyDescriptorSetLayout(device_,setLayout_,nullptr);
        vkDestroyPipeline(device_,pipeline_,nullptr);vkDestroyPipelineLayout(device_,pipeLayout_,nullptr);
        vkDestroyCommandPool(device_,commandPool_,nullptr);
        vkDestroyDevice(device_,nullptr);vkDestroyInstance(instance_,nullptr);
        std::cout<<"✅ 清理完成。\n";
    }
};

int main()
{
    std::cout<<"═══════════════════════════════════════════════════════════════════\n";
    std::cout<<" 第28章：Subgroup Operations（Wave Intrinsics）\n";
    std::cout<<"\n";
    std::cout<<" 演示：1M 个 float 的并行规约求和\n";
    std::cout<<"   Level 1: subgroupAdd()   → 硬件指令，零开销\n";
    std::cout<<"   Level 2: shared memory   → workgroup 内合并\n";
    std::cout<<"   Level 3: CPU reduce      → workgroup 间合并\n";
    std::cout<<"═══════════════════════════════════════════════════════════════════\n\n";
    Ch28App app;
    try{app.run();}catch(const std::exception&e){std::cerr<<"❌ "<<e.what()<<"\n";return 1;}
    return 0;
}
