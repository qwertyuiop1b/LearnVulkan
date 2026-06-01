/**
 * @file ch81_descriptor_buffer.cpp
 * @brief 第81章：Descriptor Buffer（VK_EXT_descriptor_buffer）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【VK_EXT_descriptor_buffer — 描述符缓冲区扩展】
 *
 *  Vulkan 1.3 最重要的扩展之一，彻底改变了描述符管理方式：
 *
 *  传统方式（DescriptorPool + DescriptorSet）：
 *    - 必须预先分配 Pool，指定每种描述符的最大数量
 *    - Pool 碎片化问题：频繁分配/释放会导致无法分配新 Set
 *    - 描述符只能通过 vkUpdateDescriptorSets 间接修改
 *    - 绑定开销：vkCmdBindDescriptorSets 每帧大量调用
 *
 *  Descriptor Buffer 方式：
 *    - 描述符存储在普通 GPU 缓冲区中（可直接 memcpy！）
 *    - 无需 Pool，无碎片问题
 *    - 通过 Device Address 直接绑定，支持 Bindless 架构
 *    - 描述符大小固定，布局完全透明
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>

#include <string>
#include <vector>

// ─── 模拟 VkPhysicalDeviceDescriptorBufferPropertiesEXT 的属性数据 ────────────
struct DescriptorBufferProps {
    size_t uniformBufferDescriptorSize = 16;
    size_t storageBufferDescriptorSize = 16;
    size_t sampledImageDescriptorSize = 24;
    size_t storageImageDescriptorSize = 24;
    size_t samplerDescriptorSize = 16;
    size_t combinedImageSamplerDescriptorSize = 32;
    size_t robustUniformBufferDescriptorSize = 16;
    size_t robustStorageBufferDescriptorSize = 16;
    size_t descriptorBufferOffsetAlignment = 256;
};

// ─── 传统方式的代码行数估算 ────────────────────────────────────────────────────
struct TraditionalStats {
    int linesPoolCreate = 12;
    int linesSetAllocate = 8;
    int linesSetWrite = 14;
    int totalPerSet() const {
        return linesPoolCreate + linesSetAllocate + linesSetWrite;
    }
};

struct DescBufferStats {
    int linesQuerySize = 5;
    int linesCreateBuf = 7;
    int linesWriteDesc = 6;
    int totalPerSet() const {
        return linesQuerySize + linesCreateBuf + linesWriteDesc;
    }
};

class Ch81App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.06f, 0.04f, 0.08f};
        queryDescriptorBufferSupport();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第81章：Descriptor Buffer");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第81章：Descriptor Buffer（VK_EXT_descriptor_buffer）", nullptr)) {
            ImGui::End();
            return;
        }

        if (!deviceSupported_) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "⚠  当前设备不支持 VK_EXT_descriptor_buffer 扩展");
            ImGui::TextWrapped("此扩展需要 Vulkan 1.2+ 并且驱动显式支持。\n"
                               "macOS / MoltenVK 截止目前尚未实现此扩展。\n\n"
                               "以下内容为教学性展示，所有属性值来自规格文档示例。");
            ImGui::Separator();
        }

        if (ImGui::BeginTabBar("DescBufTabs")) {
            buildTabTraditionalVsNew();
            buildTabAdvantages();
            buildTabDeviceProperties();
            buildTabCodeComparison();
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    bool deviceSupported_ = false;
    DescriptorBufferProps props_{};
    TraditionalStats tradStats_{};
    DescBufferStats newStats_{};

    void queryDescriptorBufferSupport() {
        // 检查设备是否支持 VK_EXT_descriptor_buffer
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(physDev_, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(physDev_, nullptr, &extCount, exts.data());

        for (const auto& ext : exts) {
            if (std::string(ext.extensionName) == "VK_EXT_descriptor_buffer") {
                deviceSupported_ = true;
                break;
            }
        }
        // 即使不支持，用规格文档的典型值填充演示数据
        // 真实查询需要 VkPhysicalDeviceDescriptorBufferPropertiesEXT 结构体
    }

    void buildTabTraditionalVsNew() {
        if (!ImGui::BeginTabItem("传统 Pool vs Descriptor Buffer"))
            return;

        ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "两种描述符管理方式的核心对比");
        ImGui::Separator();
        ImGui::Spacing();

        // ── 左栏：传统方式 ────────────────────────────────────────────────────
        ImGui::TextColored(ImVec4(1, 0.45f, 0.45f, 1), "❌  传统方式（DescriptorPool + DescriptorSet）");
        ImGui::TextWrapped("// ① 创建 Pool（必须预估每种描述符的最大数量）\n"
                           "VkDescriptorPoolSize sizes[] = {\n"
                           "    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,   1000},\n"
                           "    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},\n"
                           "};\n"
                           "VkDescriptorPoolCreateInfo poolCI{};\n"
                           "poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;\n"
                           "poolCI.maxSets       = 1000;\n"
                           "poolCI.poolSizeCount = 2;\n"
                           "poolCI.pPoolSizes    = sizes;\n"
                           "vkCreateDescriptorPool(device, &poolCI, nullptr, &pool);\n\n"
                           "// ② 从 Pool 分配 Set\n"
                           "VkDescriptorSetAllocateInfo ai{};\n"
                           "ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;\n"
                           "ai.descriptorPool     = pool;\n"
                           "ai.descriptorSetCount = 1;\n"
                           "ai.pSetLayouts        = &layout;\n"
                           "vkAllocateDescriptorSets(device, &ai, &set);\n\n"
                           "// ③ 通过 vkUpdateDescriptorSets 写描述符（间接操作）\n"
                           "VkDescriptorBufferInfo bufInfo{};\n"
                           "bufInfo.buffer = ubo;  bufInfo.range = sizeof(UBO);\n"
                           "VkWriteDescriptorSet write{};\n"
                           "write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;\n"
                           "write.dstSet          = set;\n"
                           "write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;\n"
                           "write.descriptorCount = 1;\n"
                           "write.pBufferInfo     = &bufInfo;\n"
                           "vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);\n"
                           "// 缺点：Pool 碎片、无法直接 CPU 读写、绑定开销大\n");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ── 右栏：新方式 ──────────────────────────────────────────────────────
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "✅  Descriptor Buffer 方式（直接内存操作）");
        ImGui::TextWrapped("// ① 查询描述符的字节大小\n"
                           "VkPhysicalDeviceDescriptorBufferPropertiesEXT props{\n"
                           "    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT\n"
                           "};\n"
                           "VkPhysicalDeviceProperties2 props2{...};\n"
                           "props2.pNext = &props;\n"
                           "vkGetPhysicalDeviceProperties2(physDev, &props2);\n"
                           "size_t uboSize = props.uniformBufferDescriptorSize;  // 通常 16 字节\n\n"
                           "// ② 创建描述符缓冲区（普通 GPU Buffer！）\n"
                           "VkBufferCreateInfo bufCI{};\n"
                           "bufCI.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT\n"
                           "            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;\n"
                           "bufCI.size  = uboSize * MAX_DESCRIPTORS;\n"
                           "vkCreateBuffer(device, &bufCI, nullptr, &descBuf);\n\n"
                           "// ③ 写描述符（直接 memcpy！无需 vkUpdateDescriptorSets）\n"
                           "VkDescriptorAddressInfoEXT addrInfo{\n"
                           "    VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT\n"
                           "};\n"
                           "addrInfo.address = getBufferDeviceAddress(ubo);\n"
                           "addrInfo.range   = sizeof(UBO);\n"
                           "VkDescriptorGetInfoEXT getInfo{\n"
                           "    VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT\n"
                           "};\n"
                           "getInfo.type          = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;\n"
                           "getInfo.data.pUniformBuffer = &addrInfo;\n"
                           "void* descPtr = mappedDescBuf + slot * uboSize;\n"
                           "vkGetDescriptorEXT(device, &getInfo, uboSize, descPtr);\n"
                           "// 就是这么简单！描述符就是内存，CPU 可以直接读写\n");

        ImGui::EndTabItem();
    }

    void buildTabAdvantages() {
        if (!ImGui::BeginTabItem("优势分析"))
            return;

        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1), "VK_EXT_descriptor_buffer 的核心优势");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "1. 彻底消除 Pool 管理");
        ImGui::BulletText("传统 Pool：必须预估最大描述符数量，用完后无法追加");
        ImGui::BulletText("Pool 碎片：频繁分配/释放会导致后续分配失败");
        ImGui::BulletText("Descriptor Buffer：像 malloc 一样随用随取，无碎片");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "2. CPU 可直接修改描述符");
        ImGui::BulletText("传统：必须通过 vkUpdateDescriptorSets（驱动内部操作）");
        ImGui::BulletText("Descriptor Buffer：直接 memcpy 到 mapped 内存，立即生效");
        ImGui::BulletText("调试更方便：可以读回描述符内存检查内容");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "3. 真正的 Bindless 支持");
        ImGui::BulletText("通过 VkDeviceAddress 直接绑定整个描述符缓冲区");
        ImGui::BulletText("Shader 端用 Buffer Device Address 访问任意描述符");
        ImGui::BulletText("无需分组成 DescriptorSet，完全线性地址空间");
        ImGui::TextWrapped("    // 绑定方式（替代 vkCmdBindDescriptorSets）：\n"
                           "    VkDescriptorBufferBindingInfoEXT binding{};\n"
                           "    binding.address = getBufferDeviceAddress(descBuf);\n"
                           "    binding.usage   = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;\n"
                           "    vkCmdBindDescriptorBuffersEXT(cmd, 1, &binding);\n"
                           "    // 然后在 draw call 前设置偏移即可\n"
                           "    vkCmdSetDescriptorBufferOffsetsEXT(cmd, ...);\n");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "4. 大型引擎的内存管理优势");
        ImGui::BulletText("预分配一块固定大小的描述符内存（如 64MB）");
        ImGui::BulletText("所有 Material 的描述符都在这块内存里线性排列");
        ImGui::BulletText("帧内更新：只改修改的槽位，其余不变");
        ImGui::BulletText("GPU 端可以直接生成描述符（DGC 扩展配合使用）");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "⚡ 性能对比（来自驱动厂商测试数据）");
        ImGui::BulletText("描述符更新延迟：减少约 30-50%%（无需驱动内部同步）");
        ImGui::BulletText("内存占用：通常减少 20-40%%（无 Pool 元数据开销）");
        ImGui::BulletText("CPU 端 overhead：近乎消除（只剩 memcpy 开销）");

        ImGui::EndTabItem();
    }

    void buildTabDeviceProperties() {
        if (!ImGui::BeginTabItem("设备属性查询"))
            return;

        ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1), "VkPhysicalDeviceDescriptorBufferPropertiesEXT");
        ImGui::TextWrapped("描述符 = 固定大小的内存块，大小由驱动决定。\n"
                           "通过 vkGetPhysicalDeviceProperties2 查询实际字节数：");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextWrapped("VkPhysicalDeviceDescriptorBufferPropertiesEXT props{\n"
                           "    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT\n"
                           "};\n"
                           "VkPhysicalDeviceProperties2 props2{\n"
                           "    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2\n"
                           "};\n"
                           "props2.pNext = &props;\n"
                           "vkGetPhysicalDeviceProperties2(physicalDevice, &props2);");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1), "查询结果（规格文档典型值）：");
        ImGui::Spacing();

        auto showProp = [](const char* name, size_t value, const char* note) {
            ImGui::Text("  %-50s = %3zu 字节    %s", name, value, note);
        };

        showProp("uniformBufferDescriptorSize", props_.uniformBufferDescriptorSize, "UBO 描述符");
        showProp("storageBufferDescriptorSize", props_.storageBufferDescriptorSize, "SSBO 描述符");
        showProp("sampledImageDescriptorSize", props_.sampledImageDescriptorSize, "采样纹理描述符");
        showProp("storageImageDescriptorSize", props_.storageImageDescriptorSize, "存储图像描述符");
        showProp("samplerDescriptorSize", props_.samplerDescriptorSize, "采样器描述符");
        showProp("combinedImageSamplerDescriptorSize", props_.combinedImageSamplerDescriptorSize, "组合图像采样器");
        showProp("robustUniformBufferDescriptorSize", props_.robustUniformBufferDescriptorSize, "健壮 UBO 描述符");
        showProp("robustStorageBufferDescriptorSize", props_.robustStorageBufferDescriptorSize, "健壮 SSBO 描述符");
        showProp("descriptorBufferOffsetAlignment", props_.descriptorBufferOffsetAlignment, "缓冲区偏移对齐");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1, 0.85f, 0.3f, 1), "这意味着什么？");
        ImGui::BulletText("UBO 描述符只有 16 字节：就是一个 {address, range} 结构");
        ImGui::BulletText("采样图像描述符 24 字节：包含图像视图句柄 + 采样参数");
        ImGui::BulletText("描述符不再神秘，它只是固定大小的内存块！");
        ImGui::BulletText("设计 Bindless 系统时，可以精确计算缓冲区大小：");
        ImGui::TextWrapped("    // 例如：存放 10000 个 UBO 描述符需要多少内存？\n"
                           "    size_t totalSize = 10000 * props.uniformBufferDescriptorSize;\n"
                           "    // = 10000 * 16 = 160,000 字节 ≈ 156 KB\n"
                           "    // 这就是整个描述符缓冲区的大小！\n");

        ImGui::EndTabItem();
    }

    void buildTabCodeComparison() {
        if (!ImGui::BeginTabItem("代码量对比"))
            return;

        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1), "每个描述符集的代码量对比");
        ImGui::Separator();
        ImGui::Spacing();

        // ── 进度条可视化 ──────────────────────────────────────────────────────
        float tradTotal = static_cast<float>(tradStats_.totalPerSet());
        float newTotal = static_cast<float>(newStats_.totalPerSet());
        float maxLines = std::max(tradTotal, newTotal);

        ImGui::TextColored(ImVec4(1, 0.45f, 0.45f, 1), "传统 DescriptorPool 方式");
        ImGui::Text("  创建 Pool      : %d 行", tradStats_.linesPoolCreate);
        ImGui::ProgressBar(float(tradStats_.linesPoolCreate) / maxLines,
                           ImVec2(400, 16),
                           std::to_string(tradStats_.linesPoolCreate).c_str());
        ImGui::Text("  分配 Set       : %d 行", tradStats_.linesSetAllocate);
        ImGui::ProgressBar(float(tradStats_.linesSetAllocate) / maxLines,
                           ImVec2(400, 16),
                           std::to_string(tradStats_.linesSetAllocate).c_str());
        ImGui::Text("  写描述符       : %d 行", tradStats_.linesSetWrite);
        ImGui::ProgressBar(float(tradStats_.linesSetWrite) / maxLines,
                           ImVec2(400, 16),
                           std::to_string(tradStats_.linesSetWrite).c_str());
        ImGui::TextColored(ImVec4(1, 0.45f, 0.45f, 1), "合计：%d 行/描述符集", tradStats_.totalPerSet());

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "Descriptor Buffer 方式");
        ImGui::Text("  查询描述符大小 : %d 行", newStats_.linesQuerySize);
        ImGui::ProgressBar(float(newStats_.linesQuerySize) / maxLines,
                           ImVec2(400, 16),
                           std::to_string(newStats_.linesQuerySize).c_str());
        ImGui::Text("  创建缓冲区     : %d 行", newStats_.linesCreateBuf);
        ImGui::ProgressBar(float(newStats_.linesCreateBuf) / maxLines,
                           ImVec2(400, 16),
                           std::to_string(newStats_.linesCreateBuf).c_str());
        ImGui::Text("  写描述符       : %d 行", newStats_.linesWriteDesc);
        ImGui::ProgressBar(float(newStats_.linesWriteDesc) / maxLines,
                           ImVec2(400, 16),
                           std::to_string(newStats_.linesWriteDesc).c_str());
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1),
                           "合计：%d 行/描述符集（减少 %.0f%%）",
                           newStats_.totalPerSet(),
                           (1.0f - newTotal / tradTotal) * 100.0f);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1, 0.85f, 0.3f, 1), "架构层面的影响");
        ImGui::BulletText("无需 Pool：资源管理系统少一个抽象层");
        ImGui::BulletText("无需 Set 分配追踪：内存管理统一为线性地址空间");
        ImGui::BulletText("多线程友好：各线程写自己的内存偏移，无竞争");
        ImGui::BulletText("帧间复用：未变化的描述符槽位无需任何操作");
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "支持此扩展的硬件：NVIDIA RTX 系列（驱动 525+）、AMD RDNA2/3");
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "macOS / MoltenVK：暂不支持（Metal 无对应原语）");

        ImGui::EndTabItem();
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第81章：Descriptor Buffer（VK_EXT_descriptor_buffer）\n";
    std::cout << " Vulkan 现代扩展系列 — ch81\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch81App app;
        app.run("第81章：Descriptor Buffer", 960, 720);
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
