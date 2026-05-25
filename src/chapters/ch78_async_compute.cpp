/**
 * @file ch78_async_compute.cpp
 * @brief 第78章：Async Compute（异步计算队列）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【基本概念】
 *
 *  传统 GPU 执行模型：
 *    Graphics 指令 → 顺序执行 → Compute 指令
 *    即：必须等待 Graphics 完成，才能开始 Compute
 *
 *  现代 GPU 硬件架构：
 *    · 独立的 Graphics Pipeline（VS+RS+PS）
 *    · 独立的 Compute Unit（CU / Shader Array）
 *    · 两者可以同时运行，互不干扰
 *
 *  Async Compute：
 *    · 将 Graphics 和 Compute 指令提交到不同队列
 *    · GPU 可以让两个队列并行执行
 *    · 利用 Graphics 渲染时空闲的 Compute 单元
 *
 * 【同步机制：Timeline Semaphore】
 *
 *  VkSemaphoreTypeCreateInfo::semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE
 *  · 不是 signal/wait 二进制状态，而是单调递增的计数器
 *  · 可以精确指定等待哪个"帧"的信号
 *  · 跨队列同步不需要 vkQueueWaitIdle
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <array>
#include <iostream>

class Ch78App : public DemoApp {
protected:
    void onInit() override
    {
        bgColor_ = {0.05f, 0.04f, 0.08f};
    }

    void onUpdate() override
    {
        frameTime_ += 0.016f;
        updateTimeline();
    }

    void buildUi() override
    {
        interactive_.buildDebugPanel("第78章：Async Compute");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第78章：Async Compute（异步计算队列）", nullptr))
        { ImGui::End(); return; }

        if (ImGui::BeginTabBar("AsyncTabs")) {

            // ── Tab 1: 基本概念 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("基本概念")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Async Compute 基本概念");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "GPU 硬件结构（AMD RDNA / NVIDIA Ampere）：\n\n"
                    "  ┌─────────────────────────────────────────────┐\n"
                    "  │  Graphics Engine      │  Compute Engine     │\n"
                    "  │  ─────────────────    │  ─────────────────  │\n"
                    "  │  Vertex Shader        │  Compute Shader     │\n"
                    "  │  Rasterizer           │  (同样的 CU 单元)   │\n"
                    "  │  Pixel Shader         │  但独立调度         │\n"
                    "  └─────────────────────────────────────────────┘\n\n"
                    "关键洞察：\n"
                    "  · 渲染 Shadow Map 时，Pixel Shader 负载轻\n"
                    "  → 此时 Compute Engine 是空闲的\n"
                    "  → Async Compute 可以利用这些空闲 CU\n\n"
                    "Vulkan 支持：\n"
                    "  · 需要独立的 Compute Queue（queueFamilyIndex 不同于 Graphics）\n"
                    "  · vkGetDeviceQueue(device, computeFamily, 0, &computeQueue)\n"
                    "  · 独立提交：vkQueueSubmit(computeQueue, ...)");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "并非所有 GPU 都有独立 Compute Queue：");
                ImGui::TextWrapped(
                    "  · AMD：通常有独立 Compute Queue（DMA 队列也独立）\n"
                    "  · NVIDIA：也支持，但同一家族 QueueFamily 可能有 Async Compute\n"
                    "  · 查询：vkGetPhysicalDeviceQueueFamilyProperties\n"
                    "  · 若无独立队列，可回退到串行执行（无性能提升但功能正确）");
                ImGui::EndTabItem();
            }

            // ── Tab 2: 同步机制 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("Timeline Semaphore")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Timeline Semaphore 跨队列同步");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "// 创建 Timeline Semaphore：\n"
                    "VkSemaphoreTypeCreateInfo stci{};\n"
                    "stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;\n"
                    "stci.initialValue  = 0;\n"
                    "VkSemaphoreCreateInfo sci{};\n"
                    "sci.pNext = &stci;\n"
                    "vkCreateSemaphore(device, &sci, nullptr, &timelineSem);\n\n"
                    "// Graphics Queue 完成 Shadow Map → signal(1)：\n"
                    "VkTimelineSemaphoreSubmitInfo tsi{};\n"
                    "uint64_t signalVal = 1;\n"
                    "tsi.signalSemaphoreValueCount = 1;\n"
                    "tsi.pSignalSemaphoreValues    = &signalVal;\n"
                    "submitInfo.pNext = &tsi;\n"
                    "vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);\n\n"
                    "// Compute Queue 等待 signal(1) 后才开始：\n"
                    "uint64_t waitVal = 1;\n"
                    "tsi.waitSemaphoreValueCount = 1;\n"
                    "tsi.pWaitSemaphoreValues    = &waitVal;\n"
                    "vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);\n\n"
                    "// Graphics Queue 等待 Compute 完成 signal(2)：\n"
                    "uint64_t waitCompute = 2;\n"
                    "// ... (下一帧开始时等待)");
                ImGui::EndTabItem();
            }

            // ── Tab 3: 适合任务 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("适合 Async 的任务")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "哪些任务适合 Async Compute");
                ImGui::Separator();

                struct TaskEntry {
                    const char* task;
                    const char* why;
                    const char* dependency;
                    bool        recommended;
                };
                std::array<TaskEntry, 8> tasks = {{
                    {"Shadow Map 渲染",    "深度写入，Pixel Shader 轻",  "无（可与主渲染并行）", true},
                    {"粒子物理更新",       "纯 Compute，无图形依赖",     "无（独立系统）",       true},
                    {"SSAO 计算",          "全屏 Compute，读取 G-Buffer","等待 G-Buffer Pass",   true},
                    {"SSGI 模糊",          "Compute 密集，带宽敏感",     "等待 SSGI Pass",       true},
                    {"皮肤动画蒙皮",       "纯 Compute，提前计算",       "无（提前1帧也可）",    true},
                    {"场景裁剪/剔除",      "Compute，减少 Draw Call",    "等待相机数据",         true},
                    {"G-Buffer 渲染",      "高像素填充率，占满 GPU",     "与 Compute 冲突",      false},
                    {"最终合成 Pass",      "依赖大量前序结果",           "必须等待所有Pass",     false},
                }};

                ImGui::Columns(4, "taskCols");
                ImGui::Text("任务");         ImGui::NextColumn();
                ImGui::Text("为何适合");     ImGui::NextColumn();
                ImGui::Text("依赖关系");     ImGui::NextColumn();
                ImGui::Text("推荐");         ImGui::NextColumn();
                ImGui::Separator();
                for (auto& t : tasks) {
                    if (t.recommended)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f,1,0.5f,1));
                    else
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1,0.5f,0.5f,1));
                    ImGui::Text("%s", t.task);       ImGui::NextColumn();
                    ImGui::Text("%s", t.why);        ImGui::NextColumn();
                    ImGui::Text("%s", t.dependency); ImGui::NextColumn();
                    ImGui::Text("%s", t.recommended ? "✓" : "✗"); ImGui::NextColumn();
                    ImGui::PopStyleColor();
                }
                ImGui::Columns(1);
                ImGui::EndTabItem();
            }

            // ── Tab 4: 时序图 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("时序图")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "帧时序对比（用 DrawList 绘制）");
                ImGui::Separator();

                ImGui::Checkbox("使用 Async Compute", &useAsync_);
                ImGui::SliderFloat("Graphics 帧耗时 (ms)", &graphicsMs_,  4.0f, 20.0f, "%.1f");
                ImGui::SliderFloat("Compute 任务耗时 (ms)", &computeMs_,  2.0f, 10.0f, "%.1f");
                ImGui::Spacing();

                drawTimeline();

                ImGui::Spacing();
                ImGui::Separator();
                float asyncSaving = useAsync_
                    ? std::max(0.0f, computeMs_ - (graphicsMs_ - computeMs_ * 0.5f))
                    : 0.0f;
                float totalSync  = graphicsMs_ + computeMs_;
                float totalAsync = useAsync_
                    ? std::max(graphicsMs_, computeMs_ * 0.5f + graphicsMs_ * 0.5f)
                    : totalSync;
                ImGui::Text("串行执行总耗时 : %.1f ms", totalSync);
                ImGui::Text("Async 执行总耗时 : %.1f ms", totalAsync);
                ImGui::Text("节省时间       : ~%.1f ms (%.0f%%)",
                    totalSync - totalAsync,
                    (totalSync - totalAsync) / totalSync * 100.0f);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

private:
    float frameTime_  = 0.0f;
    float graphicsMs_ = 12.0f;
    float computeMs_  = 5.0f;
    bool  useAsync_   = true;

    // 时序动画状态
    float animOffset_ = 0.0f;

    void updateTimeline()
    {
        animOffset_ = std::fmod(frameTime_ * 0.3f, 1.0f);
    }

    void drawTimeline()
    {
        ImDrawList* draw = ImGui::GetWindowDrawList();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        float width  = ImGui::GetContentRegionAvail().x - 20.0f;
        float height = useAsync_ ? 80.0f : 50.0f;
        float scale  = width / (graphicsMs_ + computeMs_ + 2.0f);

        // 背景
        draw->AddRectFilled(origin,
            ImVec2(origin.x + width, origin.y + height),
            IM_COL32(30, 30, 40, 255), 4.0f);

        float rowH = useAsync_ ? 30.0f : 24.0f;
        float y0 = origin.y + 8.0f;

        // Graphics Queue
        draw->AddText(origin, IM_COL32(200,200,200,255), "Graphics");
        float gxStart = origin.x + 80.0f;
        float gxEnd   = gxStart + graphicsMs_ * scale;
        draw->AddRectFilled(ImVec2(gxStart, y0),
            ImVec2(gxEnd, y0 + rowH),
            IM_COL32(80, 140, 220, 220), 3.0f);
        draw->AddText(ImVec2(gxStart + 4, y0 + 6), IM_COL32(255,255,255,255),
            "Graphics Rendering");

        if (useAsync_) {
            // Compute Queue（并行）
            float cxStart = gxStart + graphicsMs_ * scale * 0.3f;
            float cxEnd   = cxStart + computeMs_ * scale;
            float y1 = y0 + rowH + 8.0f;
            draw->AddText(ImVec2(origin.x, y1), IM_COL32(200,200,200,255), "Compute ");
            draw->AddRectFilled(ImVec2(cxStart, y1),
                ImVec2(cxEnd, y1 + rowH),
                IM_COL32(80, 220, 120, 200), 3.0f);
            draw->AddText(ImVec2(cxStart + 4, y1 + 6), IM_COL32(255,255,255,255),
                "Async Compute");
            // 同步线
            draw->AddLine(ImVec2(cxStart, y0), ImVec2(cxStart, y1 + rowH),
                IM_COL32(255,200,0,128), 1.5f);
        } else {
            // 串行：Compute 在 Graphics 后
            float cxStart = gxEnd + 4.0f;
            float cxEnd   = cxStart + computeMs_ * scale;
            draw->AddText(ImVec2(gxEnd + 2, y0 - 2), IM_COL32(200,200,200,255), " Compute");
            draw->AddRectFilled(ImVec2(cxStart, y0),
                ImVec2(cxEnd, y0 + rowH),
                IM_COL32(80, 220, 120, 200), 3.0f);
            draw->AddText(ImVec2(cxStart + 4, y0 + 6), IM_COL32(255,255,255,255),
                "Sequential");
        }

        ImGui::Dummy(ImVec2(width, height + 8.0f));
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第78章：Async Compute（异步计算队列）\n";
    std::cout << " 高级渲染技术系列 — ch78/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch78App app;
        app.run("第78章：Async Compute");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
