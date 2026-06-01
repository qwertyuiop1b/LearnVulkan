/**
 * @file ch82_device_generated_commands.cpp
 * @brief 第82章：设备生成命令（VK_EXT_device_generated_commands）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【VK_EXT_device_generated_commands — GPU 自主生成 Draw Call】
 *
 *  传统 GPU Driven Rendering 的瓶颈：
 *    - Compute Shader 完成视锥剔除后，写出可见物体索引
 *    - CPU 仍需执行 vkCmdBindPipeline + vkCmdBindDescriptorSets + vkCmdDraw
 *    - 不同 Material 需要切换 Pipeline → CPU 无法批量化这些命令
 *
 *  VK_EXT_device_generated_commands 的突破：
 *    - GPU 自己生成 BindPipeline / BindDescriptors / Draw 命令序列
 *    - CPU 只需调用一次 vkCmdExecuteGeneratedCommandsEXT
 *    - 真正的 "Zero CPU overhead" 渲染路径
 *
 *  与 VK_KHR_draw_indirect_count 的区别：
 *    - DrawIndirect：GPU 控制 DrawCall 数量，但 Pipeline/Descriptor 还是 CPU 绑定
 *    - DGC：GPU 控制整个命令序列，包括 Pipeline 和 Descriptor 切换
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>

#include <string>
#include <vector>

// ─── 演示用的命令序列结构体（模拟 IndirectCommandsLayout） ────────────────────
struct CommandToken {
    std::string type;
    std::string description;
    int cpuCostNs; ///< 传统方式 CPU 开销（纳秒估算）
    int gpuCostNs; ///< DGC 方式 GPU 开销（纳秒估算）
};

// ─── 场景复杂度参数 ───────────────────────────────────────────────────────────
struct SceneParams {
    int totalObjects = 10000;
    int visibleObjects = 3500;
    int materialCount = 200;
    int pipelineChanges = 150;
};

class Ch82App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.04f, 0.06f, 0.08f};
        checkDgcSupport();
        buildCommandTokens();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第82章：设备生成命令");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第82章：设备生成命令（VK_EXT_device_generated_commands）", nullptr)) {
            ImGui::End();
            return;
        }

        if (!dgcSupported_) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "⚠  当前设备不支持 VK_EXT_device_generated_commands 扩展");
            ImGui::TextWrapped("此扩展主要在高端 NVIDIA（Ampere+）和 AMD（RDNA2+）GPU 上支持。\n"
                               "macOS / MoltenVK 以及大多数移动端 GPU 暂不支持。\n\n"
                               "以下内容为教学性展示，演示 DGC 的架构思想与 API 用法。");
            ImGui::Separator();
        }

        if (ImGui::BeginTabBar("DgcTabs")) {
            buildTabPrinciple();
            buildTabUseCasesAndLimits();
            buildTabFlowComparison();
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    bool dgcSupported_ = false;
    std::vector<CommandToken> tokens_;
    SceneParams scene_{};
    int sliderObjects_ = 10000;
    int sliderMaterials_ = 200;

    void checkDgcSupport() {
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(physDev_, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(physDev_, nullptr, &extCount, exts.data());
        for (const auto& ext : exts) {
            if (std::string(ext.extensionName) == "VK_EXT_device_generated_commands") {
                dgcSupported_ = true;
                break;
            }
        }
    }

    void buildCommandTokens() {
        tokens_ = {
            {"VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT", "推送常量（Transform 矩阵等每物体数据）", 50, 5},
            {"VK_INDIRECT_COMMANDS_TOKEN_TYPE_VERTEX_BUFFER_EXT", "绑定顶点缓冲区（每 Mesh 不同）", 80, 8},
            {"VK_INDIRECT_COMMANDS_TOKEN_TYPE_INDEX_BUFFER_EXT", "绑定索引缓冲区（每 Mesh 不同）", 80, 8},
            {"VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT", "切换 Pipeline / Descriptor（每 Material）", 200, 15},
            {"VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_EXT", "执行索引绘制（真正的 Draw Call）", 30, 5},
        };
    }

    void buildTabPrinciple() {
        if (!ImGui::BeginTabItem("DGC 原理"))
            return;

        ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "GPU 自主生成命令序列的工作原理");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "传统 GPU Driven 的 CPU 瓶颈：");
        ImGui::TextWrapped("// ① Compute Shader 做视锥剔除，写出可见物体列表\n"
                           "vkCmdDispatch(cmd, objectCount / 64, 1, 1);\n\n"
                           "// ② CPU 必须等待（或用 fence/barrier）读回结果\n"
                           "vkQueueWaitIdle(queue);\n"
                           "uint32_t* visible = (uint32_t*)mappedVisibilityBuffer;\n\n"
                           "// ③ CPU 循环处理每个可见物体（瓶颈！）\n"
                           "for (uint32_t i = 0; i < visibleCount; ++i) {\n"
                           "    uint32_t objId = visible[i];\n"
                           "    Material& mat  = materials[objects[objId].matId];\n"
                           "    // 每次 Material 变化都要重新绑定：\n"
                           "    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mat.pipeline);\n"
                           "    vkCmdBindDescriptorSets(cmd, ..., mat.descriptorSet, ...);\n"
                           "    vkCmdDrawIndexed(cmd, objects[objId].indexCount, 1, ...);\n"
                           "}  // 10000 个物体 → 10000 次 CPU 函数调用！\n");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "DGC 方式：GPU 全程自主，CPU 只调用一次：");
        ImGui::TextWrapped("// ① 预先定义命令序列模板（IndirectCommandsLayout）\n"
                           "//    告诉驱动每个[命令槽]的格式\n"
                           "VkIndirectCommandsLayoutTokenEXT tokens[] = {\n"
                           "    {VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT, ...}, // Pipeline\n"
                           "    {VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT, ...}, // Transform\n"
                           "    {VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_EXT,  ...}, // Draw\n"
                           "};\n"
                           "VkIndirectCommandsLayoutCreateInfoEXT layoutCI{};\n"
                           "layoutCI.tokenCount = 3;\n"
                           "layoutCI.pTokens    = tokens;\n"
                           "vkCreateIndirectCommandsLayoutEXT(device, &layoutCI, nullptr, &cmdLayout);\n\n"
                           "// ② Compute Shader 根据剔除结果，直接向命令缓冲区写入数据\n"
                           "//    （Shader 端写：pipeline handle + push_constant + draw args）\n\n"
                           "// ③ CPU 只需一次调用执行所有生成的命令\n"
                           "vkCmdPreprocessGeneratedCommandsEXT(cmd, &preprocessInfo, stateCmd);\n"
                           "// 插入 pipeline barrier\n"
                           "vkCmdExecuteGeneratedCommandsEXT(cmd, VK_FALSE, &executeInfo);\n"
                           "// 完成！无论多少物体，CPU 开销固定不变\n");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1), "命令序列中的 Token 类型：");
        for (const auto& token : tokens_) {
            ImGui::BulletText("%-55s — %s", token.type.c_str(), token.description.c_str());
        }

        ImGui::EndTabItem();
    }

    void buildTabUseCasesAndLimits() {
        if (!ImGui::BeginTabItem("使用场景与限制"))
            return;

        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1), "DGC 的理想使用场景");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "✅ 适合的场景：");
        ImGui::BulletText("开放世界大地图：数万棵树/建筑/草，每帧 GPU 剔除后批量渲染");
        ImGui::BulletText("多 Material 场景：每个物体不同材质，需要频繁切换 Pipeline");
        ImGui::BulletText("粒子系统：GPU 端产生、销毁、渲染，无需 CPU 干预");
        ImGui::BulletText("LOD 自动切换：GPU 根据距离选择不同 Mesh + Pipeline");
        ImGui::BulletText("光线追踪 + 光栅化混合：GPU 决定哪些物体走哪条路径");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "❌ 不适合的场景：");
        ImGui::BulletText("场景中物体数量少（< 1000）：DGC 本身有 overhead");
        ImGui::BulletText("每帧物体数量变化极大：预处理缓冲区大小难以确定");
        ImGui::BulletText("需要精确 CPU 端绘制顺序控制（透明物体排序等）");
        ImGui::Spacing();

        ImGui::Separator();
        ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "硬件支持情况：");
        ImGui::Spacing();

        struct HardwareEntry {
            const char* hardware;
            const char* support;
            ImVec4 color;
        };
        HardwareEntry entries[] = {
            {"NVIDIA RTX 3000/4000/5000 系列（Ampere/Ada/Blackwell）", "✅ 完整支持", {0.3f, 1.0f, 0.5f, 1}},
            {"AMD RX 6000/7000 系列（RDNA2/RDNA3）", "✅ 完整支持", {0.3f, 1.0f, 0.5f, 1}},
            {"Intel Arc A 系列（Alchemist）", "⚠ 部分支持（驱动版本相关）", {1, 0.85f, 0.3f, 1}},
            {"Apple Silicon（M1/M2/M3/M4）/ MoltenVK", "❌ 不支持（Metal 无对应原语）", {1, 0.4f, 0.4f, 1}},
            {"移动端 GPU（Adreno/Mali）", "❌ 通常不支持", {1, 0.4f, 0.4f, 1}},
        };
        for (const auto& e : entries) {
            ImGui::TextColored(e.color, "  %-55s %s", e.hardware, e.support);
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "API 约束（开发者需注意）：");
        ImGui::BulletText("IndirectCommandsLayout 在 Pipeline 创建后才能确定");
        ImGui::BulletText("需要预先分配足够大的预处理缓冲区（vkGetGeneratedCommandsMemoryRequirementsEXT）");
        ImGui::BulletText("命令生成 Compute Shader 必须知道最大命令数上限");
        ImGui::BulletText("调试困难：GPU 生成的命令在 RenderDoc 中需要专门支持");

        ImGui::EndTabItem();
    }

    void buildTabFlowComparison() {
        if (!ImGui::BeginTabItem("流程对比与性能估算"))
            return;

        ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "每帧渲染流程对比");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::SliderInt("场景物体总数", &sliderObjects_, 100, 50000);
        ImGui::SliderInt("材质种类数量", &sliderMaterials_, 1, 500);
        ImGui::Spacing();

        scene_.totalObjects = sliderObjects_;
        scene_.visibleObjects = sliderObjects_ * 35 / 100;
        scene_.materialCount = sliderMaterials_;
        scene_.pipelineChanges = sliderMaterials_ * 75 / 100;

        // ── 传统流程 ──────────────────────────────────────────────────────────
        ImGui::TextColored(ImVec4(1, 0.45f, 0.45f, 1), "传统流程（CPU 主导）：");
        ImGui::Text("  ① GPU 剔除 Compute Dispatch        → GPU 执行（异步）");
        ImGui::Text("  ② CPU 等待 Fence / Barrier          → CPU 阻塞");
        ImGui::Text("  ③ CPU 遍历 %5d 个可见物体         → CPU 循环", scene_.visibleObjects);
        ImGui::Text(
            "  ④ CPU vkCmdBindPipeline × %4d 次  → %6d 次驱动调用", scene_.pipelineChanges, scene_.pipelineChanges);
        ImGui::Text(
            "  ⑤ CPU vkCmdDraw          × %4d 次  → %6d 次驱动调用", scene_.visibleObjects, scene_.visibleObjects);
        int tradCpuCalls = scene_.pipelineChanges * 3 + scene_.visibleObjects;
        ImGui::TextColored(ImVec4(1, 0.45f, 0.45f, 1), "  CPU 驱动调用总次数：约 %d 次/帧", tradCpuCalls);

        float tradMs = float(tradCpuCalls) * 0.005f;
        ImGui::Text("  CPU 端估算耗时：%.2f ms（@5ns/调用）", tradMs);
        ImGui::ProgressBar(tradMs / 16.0f, ImVec2(400, 18));
        ImGui::TextDisabled("  （1/60 帧预算 = 16.67ms）");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ── DGC 流程 ──────────────────────────────────────────────────────────
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "DGC 流程（GPU 全程自主）：");
        ImGui::Text("  ① GPU 剔除 Compute Dispatch        → GPU 执行（异步）");
        ImGui::Text("  ② GPU 写命令缓冲区（Compute Shader） → GPU 内部，无 CPU 参与");
        ImGui::Text("  ③ vkCmdPreprocessGeneratedCommandsEXT → CPU 1 次调用");
        ImGui::Text("  ④ vkCmdExecuteGeneratedCommandsEXT    → CPU 1 次调用");
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "  CPU 驱动调用总次数：固定 2 次/帧（无论物体数量！）");

        float dgcMs = 0.01f;
        ImGui::Text("  CPU 端估算耗时：< %.2f ms（固定开销）", dgcMs);
        ImGui::ProgressBar(dgcMs / 16.0f, ImVec2(400, 18));

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float speedup = tradMs / std::max(dgcMs, 0.001f);
        ImGui::TextColored(ImVec4(1, 0.85f, 0.3f, 1), "理论 CPU 端提速：%.0fx  （物体越多，优势越大）", speedup);
        ImGui::Spacing();

        // ── 命令 Token CPU vs GPU 耗时对比表 ─────────────────────────────────
        ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1), "各 Token 类型耗时对比：");
        ImGui::Separator();
        if (ImGui::BeginTable("TokenTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Token 类型");
            ImGui::TableSetupColumn("说明");
            ImGui::TableSetupColumn("CPU 耗时（ns/次）");
            ImGui::TableSetupColumn("GPU 耗时（ns/次）");
            ImGui::TableHeadersRow();
            for (const auto& t : tokens_) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(t.type.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(t.description.c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%d", t.cpuCostNs);
                ImGui::TableNextColumn();
                ImGui::Text("%d", t.gpuCostNs);
            }
            ImGui::EndTable();
        }

        ImGui::EndTabItem();
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第82章：设备生成命令（VK_EXT_device_generated_commands）\n";
    std::cout << " Vulkan 现代扩展系列 — ch82\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch82App app;
        app.run("第82章：Device Generated Commands", 960, 720);
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
