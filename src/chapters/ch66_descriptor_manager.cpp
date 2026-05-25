/**
 * @file ch66_descriptor_manager.cpp
 * @brief 第66章：描述符集管理（DescriptorAllocator / DescriptorBuilder）
 *
 * 【三个核心问题】
 *  1. DescriptorPool 容量不够（OUT_OF_POOL_MEMORY）
 *     → DescriptorAllocator：自动扩容，永不失败
 *
 *  2. DescriptorSetLayout 重复创建（相同绑定创建多次）
 *     → DescriptorLayoutCache：哈希去重
 *
 *  3. vkUpdateDescriptorSets 样板代码（每个 binding 5 行）
 *     → DescriptorBuilder：流式绑定，1 行 = 1 个 binding
 *
 * 【DescriptorBuilder 用法】
 *  VkDescriptorSet set;
 *  VkDescriptorSetLayout layout;
 *  DescriptorBuilder(allocator, layoutCache)
 *      .bindBuffer(0, cameraUBO.descriptorInfo(fi))
 *      .bindImage(1, albedoTex.descriptorInfo())
 *      .bindImage(2, normalTex.descriptorInfo())
 *      .bindStorageBuffer(3, instanceSSBO.descriptorInfo())
 *      .build(set, layout);   // ← 分配 + 写入一步完成
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <vulkan_tutorial/engine/descriptor_manager.hpp>

class Ch66App : public DemoApp {
protected:
    void onInit() override
    {
        bgColor_ = {0.04f, 0.08f, 0.12f};
        // 模拟统计（实际引擎中会传入 RHIDevice& 初始化）
        simulate();
    }

    void onShutdown() override {}


    void buildUi() override
    {
        interactive_.buildDebugPanel("第66章：描述符集管理");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 690), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Descriptor 封装 — Allocator / LayoutCache / Builder", nullptr))
        { ImGui::End(); return; }

        if (ImGui::BeginTabBar("DescTabs")) {

            if (ImGui::BeginTabItem("DescriptorBuilder API")) {
                ImGui::TextColored(ImVec4(1,0.85f,0.2f,1), "DescriptorBuilder — 流式描述符构建");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "// 传统写法（30 行 / set）：\n"
                    "VkDescriptorSetLayoutBinding b0{};\n"
                    "b0.binding = 0; b0.descriptorType = UNIFORM_BUFFER; ...\n"
                    "VkDescriptorSetLayoutCreateInfo lci{};\n"
                    "lci.bindingCount = 2; lci.pBindings = bindings;\n"
                    "VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, ...));\n"
                    "VkDescriptorSetAllocateInfo ai{}; ...\n"
                    "VK_CHECK(vkAllocateDescriptorSets(device, &ai, &set));\n"
                    "VkDescriptorBufferInfo bi{}; bi.buffer = ubo; bi.range = size;\n"
                    "VkWriteDescriptorSet w{}; w.dstBinding = 0; w.pBufferInfo = &bi; ...\n"
                    "vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);\n\n"
                    "// DescriptorBuilder（4 行）：\n"
                    "VkDescriptorSet set;\n"
                    "VkDescriptorSetLayout layout;\n"
                    "DescriptorBuilder(allocator, layoutCache)\n"
                    "    .bindBuffer(0, cameraUBO.descriptorInfo(fi))\n"
                    "    .bindImage(1, albedoTex.descriptorInfo())\n"
                    "    .build(set, layout);\n\n"
                    "// 下次创建相同 layout：\n"
                    "DescriptorBuilder(allocator, layoutCache)\n"
                    "    .bindBuffer(0, anotherUBO.descriptorInfo(fi))\n"
                    "    .bindImage(1, anotherTex.descriptorInfo())\n"
                    "    .build(set2, layout2);  // layout2 == layout（缓存命中）\n");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("DescriptorAllocator")) {
                ImGui::TextColored(ImVec4(0.4f,1,0.5f,1), "DescriptorAllocator — 自动扩容 Pool");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "// 传统写法痛点：\n"
                    "// 需要提前估算每种描述符类型的数量，创建一个大 Pool\n"
                    "// 估算不准：OOM → 崩溃；估算过大 → 浪费显存\n\n"
                    "// DescriptorAllocator 策略：\n"
                    "// 每个 Pool 预留 1000 个/类型（约 100MB），分配失败时自动创建新 Pool\n"
                    "// 帧结束后 reset()：归还所有 set，Pool 不销毁，下帧复用\n\n");
                ImGui::TextColored(ImVec4(0.4f,0.8f,1,1), "当前 Allocator 状态");
                ImGui::Separator();
                ImGui::Text("Pool 数量       : %zu", simPoolCount_);
                ImGui::Text("已分配 Set 数   : %zu", simAllocated_);
                ImGui::Text("Layout 缓存数   : %zu", simLayoutsCached_);
                ImGui::Spacing();
                if (ImGui::Button("分配 10 个 DescriptorSet（模拟）")) {
                    simAllocated_ += 10;
                    if (simAllocated_ > simPoolCount_ * 4000) ++simPoolCount_;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset（归还所有，模拟）")) {
                    simAllocated_ = 0;
                }
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("DescriptorLayoutCache")) {
                ImGui::TextColored(ImVec4(1,0.5f,0.8f,1), "DescriptorLayoutCache — 哈希去重");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "// 问题：多个不同对象使用相同的 DescriptorSetLayout\n"
                    "// 每次都创建新的 layout 对象 → 浪费 GPU 对象 + 内存泄漏风险\n\n"
                    "// DescriptorLayoutCache 解决方案：\n"
                    "// 对 VkDescriptorSetLayoutBinding 数组进行哈希\n"
                    "// 相同内容（binding/type/stage/count 完全一致）→ 返回同一 VkDescriptorSetLayout\n\n"
                    "// 哈希函数：\n"
                    "size_t hash = 0;\n"
                    "for (auto& b : bindings) {\n"
                    "    hash ^= hash_combine(b.binding, b.descriptorType,\n"
                    "                         b.stageFlags, b.descriptorCount);\n"
                    "}\n\n");
                ImGui::Text("已缓存 Layout 数量 : %zu", simLayoutsCached_);
                ImGui::Text("（若无缓存，每次创建 100 个 Material 会创建 100 个重复 layout）");
                ImGui::Text("（有缓存，100 个 Material 只创建 1 个 layout 对象）");
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

private:
    // 模拟统计（教学演示，不需要实际 Vulkan 分配）
    size_t simPoolCount_     = 1;
    size_t simAllocated_     = 0;
    size_t simLayoutsCached_ = 0;

    void simulate()
    {
        simPoolCount_     = 1;
        simAllocated_     = 5;
        simLayoutsCached_ = 3;
    }
};

// 修复：更新 buildUi 中的引用
// （使用 simXxx_ 变量替代 descAllocator_.xxx()）

int main()
{
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第66章：描述符集管理（DescriptorAllocator / DescriptorBuilder）\n";
    std::cout << " 引擎封装系列 — ch66/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try { Ch66App app; app.run("第66章：描述符集管理"); std::cout<<"✅ 正常退出\n"; }
    catch(const std::exception& e){ std::cerr<<"❌ "<<e.what()<<"\n"; return 1; }
}
