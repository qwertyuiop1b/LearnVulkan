/**
 * @file ch67_command_recorder.cpp
 * @brief 第67章：命令录制封装（CommandRecorder / RenderPassScope / BarrierBatch / DrawCallBatch）
 *
 * 【四个组件】
 *  CommandRecorder   — vkBegin/End 的 RAII 包装，析构时自动 End
 *  RenderPassScope   — vkCmdBeginRenderPass/EndRenderPass 的 RAII 包装
 *  BarrierBatch      — 收集多个 barrier，一次 vkCmdPipelineBarrier 提交（批处理优化）
 *  DrawCallBatch     — 按 (pipeline, material, mesh) 排序 draw calls，减少状态切换
 *
 * 【DrawCallBatch 排序优化】
 *  不排序：12 个物体，3 种管线，随机顺序绘制 → 管线切换 12 次
 *  排序后：同管线连续绘制 → 管线切换仅 3 次（节省 ~75%）
 *
 * 【DrawKey 64-bit 编码】
 *  [63:56] 渲染层（不透明/透明/UI）
 *  [55:40] Pipeline ID（减少管线切换）
 *  [39:24] Material ID（减少描述符切换）
 *  [23: 0] Mesh ID
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <vulkan_tutorial/engine/command_recorder.hpp>

#include <algorithm>
#include <random>

class Ch67App : public DemoApp {
protected:
    void onInit() override
    {
        bgColor_ = {0.06f, 0.06f, 0.10f};
        generateDrawCalls();
    }

    void buildUi() override
    {
        interactive_.buildDebugPanel("第67章：命令录制封装");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 690), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("CommandRecorder — RAII 录制 + 排序优化", nullptr))
        { ImGui::End(); return; }

        if (ImGui::BeginTabBar("CmdTabs")) {

            if (ImGui::BeginTabItem("RAII 录制")) {
                ImGui::TextColored(ImVec4(1,0.85f,0.2f,1), "CommandRecorder + RenderPassScope");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "// 传统写法（容易忘记 End）：\n"
                    "VkCommandBufferBeginInfo bi{};\n"
                    "bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;\n"
                    "vkBeginCommandBuffer(cmd, &bi);\n"
                    "// ... 中间如果 throw，vkEnd 不会被调用！\n"
                    "vkEndCommandBuffer(cmd);\n\n"
                    "// CommandRecorder RAII（析构时自动 End）：\n"
                    "{\n"
                    "    CommandRecorder rec(dev, cmd, CommandRecorder::OneShot);\n"
                    "    // ... 录制命令 ...\n"
                    "    // 就算这里 throw，析构器也会调用 vkEndCommandBuffer\n"
                    "}  // ← 自动 vkEndCommandBuffer\n\n"
                    "// RenderPassScope（析构时自动 EndRenderPass）：\n"
                    "{\n"
                    "    RenderPassScope rp(cmd, renderPass, fb, extent,\n"
                    "        { clearColor, clearDepth });\n"
                    "    // ... vkCmdDraw ...\n"
                    "}  // ← 自动 vkCmdEndRenderPass\n");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("BarrierBatch")) {
                ImGui::TextColored(ImVec4(0.4f,1,0.5f,1), "BarrierBatch — 批量 Pipeline Barrier");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "// 传统写法：每个 barrier 单独调用 vkCmdPipelineBarrier（效率低）\n"
                    "transitionImage(cmd, hdrImage,   UNDEFINED → COLOR_ATTACHMENT);\n"
                    "transitionImage(cmd, depthImage, UNDEFINED → DEPTH_ATTACHMENT);\n"
                    "transitionImage(cmd, shadowMap,  READ_ONLY → DEPTH_ATTACHMENT);\n"
                    "// 上面 3 次调用 = 3 次 vkCmdPipelineBarrier\n\n"
                    "// BarrierBatch：收集后一次提交\n"
                    "{\n"
                    "    BarrierBatch batch(cmd);\n"
                    "    batch.imageLayout(hdrImage,   UNDEFINED, COLOR_ATTACHMENT);\n"
                    "    batch.imageLayout(depthImage, UNDEFINED, DEPTH_ATTACHMENT);\n"
                    "    batch.imageLayout(shadowMap,  READ_ONLY, DEPTH_ATTACHMENT);\n"
                    "}  // ← 析构时一次 vkCmdPipelineBarrier，GPU 可合并处理\n\n"
                    "// 好处：GPU 可以对同批次 barrier 做重叠优化\n"
                    "// BarrierBatch 自动根据 ImageLayout 推导 access/stage flags\n");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("DrawCallBatch 排序")) {
                ImGui::TextColored(ImVec4(1,0.5f,0.8f,1), "DrawCallBatch — 减少管线切换次数");
                ImGui::Separator();

                ImGui::Text("场景：%d 个物体，%d 种管线", (int)drawCalls_.size(), 3);
                ImGui::Separator();

                // 未排序状态
                ImGui::TextColored(ImVec4(1,0.4f,0.4f,1), "❌ 未排序（随机顺序）");
                ImGui::Text("  管线切换次数：%d / %d 次 draw call（切换率 %.0f%%）",
                    unsortedSwitches_, (int)drawCalls_.size(),
                    float(unsortedSwitches_)/drawCalls_.size()*100.0f);

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.3f,1,0.4f,1), "✅ 排序后（按 DrawKey 升序）");
                ImGui::Text("  管线切换次数：%d / %d 次 draw call（切换率 %.0f%%）",
                    sortedSwitches_, (int)drawCalls_.size(),
                    float(sortedSwitches_)/drawCalls_.size()*100.0f);

                float saving = float(unsortedSwitches_ - sortedSwitches_)/unsortedSwitches_*100.0f;
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1,1,0.3f,1),
                    "节省管线切换：%d 次（%.0f%% 减少）",
                    unsortedSwitches_ - sortedSwitches_, saving);

                ImGui::Separator();
                ImGui::TextWrapped(
                    "// DrawKey 编码（64 bit）：\n"
                    "// [63:56] Layer（0=Opaque, 1=AlphaTest, 2=Transparent）\n"
                    "// [55:40] Pipeline ID\n"
                    "// [39:24] Material ID\n"
                    "// [23: 0] Mesh ID\n\n"
                    "DrawCallBatch batch;\n"
                    "for (auto& obj : scene) {\n"
                    "    DrawCall dc;\n"
                    "    dc.key = DrawKey::make(Opaque, pipelineId, materialId, meshId);\n"
                    "    dc.pipeline = obj.pipeline;\n"
                    "    dc.descSet  = obj.descriptorSet;\n"
                    "    // ...\n"
                    "    batch.add(dc);\n"
                    "}\n"
                    "batch.sort();       // 按 key 排序\n"
                    "batch.flush(cmd);   // 最优顺序提交，减少状态切换\n");

                ImGui::Spacing();
                if (ImGui::Button("重新生成随机顺序")) {
                    generateDrawCalls();
                }
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

private:
    struct SimDrawCall { int pipelineId; int materialId; int meshId; };
    std::vector<SimDrawCall> drawCalls_;
    int unsortedSwitches_ = 0;
    int sortedSwitches_   = 0;

    void generateDrawCalls()
    {
        std::mt19937 rng(42);
        drawCalls_.clear();
        for (int i = 0; i < 24; ++i) {
            drawCalls_.push_back({int(rng()%3), int(rng()%8), int(rng()%12)});
        }
        // 计算未排序切换次数
        unsortedSwitches_ = 0;
        for (size_t i = 1; i < drawCalls_.size(); ++i)
            if (drawCalls_[i].pipelineId != drawCalls_[i-1].pipelineId)
                ++unsortedSwitches_;
        // 排序后
        auto sorted = drawCalls_;
        std::sort(sorted.begin(), sorted.end(), [](const SimDrawCall& a, const SimDrawCall& b){
            if (a.pipelineId != b.pipelineId) return a.pipelineId < b.pipelineId;
            if (a.materialId != b.materialId) return a.materialId < b.materialId;
            return a.meshId < b.meshId;
        });
        sortedSwitches_ = 0;
        for (size_t i = 1; i < sorted.size(); ++i)
            if (sorted[i].pipelineId != sorted[i-1].pipelineId)
                ++sortedSwitches_;
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第67章：命令录制封装（CommandRecorder / DrawCallBatch）\n";
    std::cout << " 引擎封装系列 — ch67/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try { Ch67App app; app.run("第67章：命令录制封装"); std::cout<<"✅ 正常退出\n"; }
    catch(const std::exception& e){ std::cerr<<"❌ "<<e.what()<<"\n"; return 1; }
}
