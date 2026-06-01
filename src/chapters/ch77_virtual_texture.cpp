/**
 * @file ch77_virtual_texture.cpp
 * @brief 第77章：虚拟纹理流式加载（Virtual Texture Streaming）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【问题背景】
 *
 *  超大地形纹理（如 16K×16K）全部加载到 GPU 显存是不可能的：
 *    16K × 16K × 4 byte/px = 1 GB 显存！
 *
 *  解决方案：虚拟纹理（Virtual Texture，又称 Sparse Texture / Megatexture）
 *
 * 【核心概念】
 *
 *  逻辑纹理（Virtual Texture）：
 *    · 2D 地址空间（如 16K×16K）
 *    · 被分成固定大小的页（Page），如 128×128 px
 *    · 逻辑上存在，物理上不占用连续显存
 *
 *  物理纹理（Physical Atlas）：
 *    · GPU 实际分配的小纹理（如 2048×2048）
 *    · 存储当前使用中的虚拟页
 *    · 相当于"显存中的页缓存"
 *
 *  页表（Page Table）：
 *    · 存储虚拟页 → 物理块的映射
 *    · 实现为一张纹理（每像素 = 一个虚拟页的物理位置）
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

static constexpr int VIRTUAL_SIZE = 16384;
static constexpr int PAGE_SIZE = 128;
static constexpr int VIRTUAL_PAGES_DIM = VIRTUAL_SIZE / PAGE_SIZE;
static constexpr int TOTAL_PAGES = VIRTUAL_PAGES_DIM * VIRTUAL_PAGES_DIM;
static constexpr int PHYSICAL_SIZE = 2048;
static constexpr int PHYSICAL_PAGES = (PHYSICAL_SIZE / PAGE_SIZE) * (PHYSICAL_SIZE / PAGE_SIZE);

class Ch77App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.04f, 0.07f, 0.05f};
        cachedPages_.resize(PHYSICAL_PAGES, false);
        requestedPages_.clear();
        simulateInitialCache();
    }

    void onUpdate() override {
        frameTime_ += 0.016f;
        if (isSimulating_) {
            simulateFrameRequests();
        }
        updateStats();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第77章：虚拟纹理");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第77章：虚拟纹理流式加载（Virtual Texture Streaming）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("VtTabs")) {

            // ── Tab 1: 系统架构 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("系统架构")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "虚拟纹理系统架构");
                ImGui::Separator();
                ImGui::TextWrapped("逻辑结构：\n"
                                   "  虚拟纹理空间：16384 × 16384 px\n"
                                   "  页面大小：128 × 128 px\n"
                                   "  虚拟页总数：128 × 128 = 16384 页\n\n"
                                   "物理分配：\n"
                                   "  物理 Atlas：2048 × 2048 px（GPU 常驻）\n"
                                   "  物理块大小：128 × 128 px\n"
                                   "  物理块总数：16 × 16 = 256 块\n\n"
                                   "页表（Page Table Texture）：\n"
                                   "  尺寸：128 × 128 px（每像素=一个虚拟页）\n"
                                   "  格式：RG8（R=物理X，G=物理Y，A=有效标志）\n"
                                   "  用途：着色器查表找到物理块位置\n\n"
                                   "着色器查找流程：\n"
                                   "  // 1. 计算虚拟页坐标\n"
                                   "  vec2 pageCoord = uv * VIRTUAL_PAGES_DIM;\n"
                                   "  vec2 pageIdx = floor(pageCoord);\n\n"
                                   "  // 2. 查页表\n"
                                   "  vec4 pageEntry = texture(pageTable, pageIdx / VIRTUAL_PAGES_DIM);\n"
                                   "  if (pageEntry.a < 0.5) { outColor = fallbackColor; return; }\n\n"
                                   "  // 3. 映射到物理 Atlas 坐标\n"
                                   "  vec2 physPagePos = pageEntry.rg * 255.0;  // 物理块位置\n"
                                   "  vec2 inPageUV = fract(pageCoord);          // 页内偏移\n"
                                   "  vec2 atlasUV = (physPagePos + inPageUV) / PHYSICAL_PAGES_DIM;\n"
                                   "  outColor = texture(physAtlas, atlasUV);");
                ImGui::EndTabItem();
            }

            // ── Tab 2: 反馈缓冲 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("反馈缓冲")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Feedback Buffer — 页面请求机制");
                ImGui::Separator();
                ImGui::TextWrapped("反馈缓冲（Feedback Buffer）工作原理：\n\n"
                                   "GPU → CPU 通信：\n"
                                   "  1. 渲染时，着色器写出当前需要的虚拟页地址：\n"
                                   "     // feedback.frag\n"
                                   "     outFeedback.rg = pageCoord / 255.0;  // 虚拟页XY\n"
                                   "     outFeedback.b  = mipLevel / 255.0;   // Mip 层级\n\n"
                                   "  2. CPU 读回 Feedback Buffer（可用 1/8 降采样版本）\n"
                                   "     · 使用 vkCmdCopyImageToBuffer 传回 CPU\n"
                                   "     · 延迟 1-2 帧读取（避免等待）\n\n"
                                   "  3. CPU 处理：\n"
                                   "     · 统计每个页面的请求频率\n"
                                   "     · 决定加载哪些页（基于优先级）\n"
                                   "     · 从磁盘/资源包异步读取页面数据\n"
                                   "     · 上传到物理 Atlas\n"
                                   "     · 更新页表纹理\n\n"
                                   "  4. 页面淘汰策略（LRU - Least Recently Used）：\n"
                                   "     · 物理块满时，淘汰最久未使用的页\n"
                                   "     · 更新页表（标记为无效）\n"
                                   "     · 下次访问该虚拟页时触发 page fault\n\n"
                                   "Page Fault 处理：\n"
                                   "  · Fallback：使用低 Mip 的近似颜色（不显示黑块）\n"
                                   "  · 异步加载正确分辨率的页面\n"
                                   "  · 加载完成后替换 Fallback");
                ImGui::EndTabItem();
            }

            // ── Tab 3: 流式队列 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("流式队列演示")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "页面流式加载模拟");
                ImGui::Separator();

                ImGui::Checkbox("开启模拟", &isSimulating_);
                ImGui::SliderFloat("相机移动速度", &cameraSpeed_, 0.0f, 20.0f, "%.1f m/s");
                ImGui::Spacing();
                ImGui::Separator();

                int cachedCount = 0;
                for (bool b : cachedPages_)
                    if (b)
                        ++cachedCount;

                float cacheUtil = float(cachedCount) / PHYSICAL_PAGES;
                ImGui::Text("物理 Atlas 大小   : %d × %d px = %d 块", PHYSICAL_SIZE, PHYSICAL_SIZE, PHYSICAL_PAGES);
                ImGui::Text("已缓存页数        : %d / %d", cachedCount, PHYSICAL_PAGES);
                ImGui::Text("物理纹理利用率    :");
                ImGui::SameLine();
                ImGui::ProgressBar(cacheUtil, ImVec2(300, 18));

                ImGui::Text("本帧请求页数      : %d", framePageRequests_);
                ImGui::Text("本帧加载完成页数  : %d", framePageLoaded_);
                ImGui::Text("流式队列待加载    : %d", streamQueueDepth_);
                ImGui::Text("Page Fault 计数   : %d", pageFaultCount_);
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "各 Mip 层级缓存情况：");
                const char* mipNames[] = {"Mip 0 (原始)", "Mip 1 (1/2)", "Mip 2 (1/4)", "Mip 3 (1/8)"};
                for (int i = 0; i < 4; ++i) {
                    float mipFill = std::min(1.0f, cacheUtil + (i * 0.15f));
                    ImGui::Text("  %s:", mipNames[i]);
                    ImGui::SameLine(200);
                    ImGui::ProgressBar(mipFill, ImVec2(200, 14));
                }
                ImGui::EndTabItem();
            }

            // ── Tab 4: 性能分析 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("性能分析")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "虚拟纹理性能特征");
                ImGui::Separator();
                ImGui::TextWrapped("显存节省：\n"
                                   "  无虚拟纹理：16K×16K×4B = 1 GB\n"
                                   "  使用虚拟纹理：物理 Atlas 2K×2K×4B = 16 MB\n"
                                   "  节省比例：98.4%%\n\n"
                                   "性能开销：\n"
                                   "  · 额外纹理采样：1次页表查询 + 1次Atlas查询\n"
                                   "  · 页表纹理：极小（128×128 = 64 KB）\n"
                                   "  · Feedback Pass：1/8 分辨率（开销很低）\n"
                                   "  · CPU 处理 Feedback：每帧约 0.5-1 ms\n\n"
                                   "局限性：\n"
                                   "  · 需要额外的着色器复杂度\n"
                                   "  · 页面加载时可能短暂出现低精度\n"
                                   "  · 不适合高频纹理（会频繁换页）\n"
                                   "  · Vulkan 的 Sparse Binding 特性可进一步优化\n\n"
                                   "Vulkan Sparse Texture：\n"
                                   "  · vkBindImageMemory2 + VkSparseImageMemoryBind\n"
                                   "  · 可以将物理内存直接绑定到虚拟纹理的某些页面\n"
                                   "  · 不需要 Atlas + 页表的间接层\n"
                                   "  · 但并非所有 GPU 都支持 sparseResidency");

                ImGui::Spacing();
                ImGui::Text("当前模拟帧 : %d", static_cast<int>(frameTime_ / 0.016f));
                ImGui::Text("总 Page Fault 次数 : %d", pageFaultCount_);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    std::vector<bool> cachedPages_;
    std::vector<int> requestedPages_;
    float frameTime_ = 0.0f;
    float cameraSpeed_ = 5.0f;
    bool isSimulating_ = true;
    int framePageRequests_ = 0;
    int framePageLoaded_ = 0;
    int streamQueueDepth_ = 0;
    int pageFaultCount_ = 0;

    void simulateInitialCache() {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, PHYSICAL_PAGES - 1);
        for (int i = 0; i < PHYSICAL_PAGES * 3 / 4; ++i) {
            cachedPages_[dist(rng)] = true;
        }
    }

    void simulateFrameRequests() {
        std::mt19937 rng(static_cast<uint32_t>(frameTime_ * 100));
        int requestCount = static_cast<int>(cameraSpeed_ * 2.5f) + 3;
        std::uniform_int_distribution<int> pageDist(0, PHYSICAL_PAGES - 1);

        framePageRequests_ = requestCount;
        framePageLoaded_ = 0;
        for (int i = 0; i < requestCount; ++i) {
            int page = pageDist(rng);
            if (!cachedPages_[page]) {
                ++pageFaultCount_;
                cachedPages_[page] = true;
                ++framePageLoaded_;
                int evict = pageDist(rng);
                cachedPages_[evict] = false;
            }
        }
        streamQueueDepth_ = std::max(0, streamQueueDepth_ + requestCount - framePageLoaded_ * 2);
    }

    void updateStats() {}
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第77章：虚拟纹理流式加载（Virtual Texture Streaming）\n";
    std::cout << " 高级渲染技术系列 — ch77/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch77App app;
        app.run("第77章：虚拟纹理");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
