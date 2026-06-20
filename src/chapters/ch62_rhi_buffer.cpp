/**
 * @file ch62_rhi_buffer.cpp
 * @brief 第62章：GPU 缓冲区封装（Buffer / VertexBuffer / UniformBuffer<T> / StagingPool）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【封装对比】
 *
 *  传统写法（每次都要写这些）：
 *    // 创建 vertex buffer：
 *    vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
 *    vkGetBufferMemoryRequirements(device, buffer, &memReqs);
 *    allocInfo.memoryTypeIndex = findMemoryType(physDev, memReqs.memoryTypeBits, props);
 *    vkAllocateMemory(device, &allocInfo, nullptr, &memory);
 *    vkBindBufferMemory(device, buffer, memory, 0);
 *    // staging + copy ...（再加 20 行）
 *
 *  Buffer 封装后：
 *    Buffer::CreateInfo ci{};
 *    ci.size  = sizeof(MyVertex) * vertices.size();
 *    ci.usage = VERTEX_BUFFER | TRANSFER_DST;
 *    buf.create(dev, ci);
 *    buf.upload(dev, vertices.data(), ci.size);   // 自动处理 staging
 *
 *  VertexBuffer<T>（更高层）：
 *    VertexBuffer vb;
 *    vb.create(dev, vertices);   // 1 行
 *
 *  UniformBuffer<CameraData>：
 *    UniformBuffer<CameraData> ubo;
 *    ubo.create(dev, MAX_FRAMES);
 *    ubo.update(frameIndex, { view, proj });  // 每帧 1 行
 *
 * 【本章 Demo】
 *   模拟 Buffer 封装的核心行为，ImGui 显示：
 *   - 缓冲区分配统计（次数、总大小）
 *   - StagingPool 命中率
 *   - UniformBuffer 更新次数/帧
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <vulkan_tutorial/engine/rhi_buffer.hpp>

#include <chrono>
#include <random>
#include <sstream>

// ─── 模拟 BufferManager 统计器 ─────────────────────────────────────────────

struct BufferStats {
    uint32_t totalAllocations = 0;
    uint32_t stagingAllocations = 0;
    uint32_t stagingCacheHits = 0;
    uint64_t totalBytesAllocated = 0;
    uint32_t activeBuffers = 0;
    uint32_t framesRendered = 0;
    uint32_t uboUpdatesThisFrame = 0;
};

class Ch62App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.04f, 0.06f, 0.10f};
        interactive_.camera().setDistance(8.0f);
        rng_.seed(42);

        // 模拟创建各种缓冲区
        simulateBufferCreation();
    }

    void onUpdate() override {
        ++stats_.framesRendered;
        // 每帧模拟 UBO 更新（相机、灯光、每物体矩阵）
        stats_.uboUpdatesThisFrame = 3 + static_cast<uint32_t>(simulatedObjects_);
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第62章：缓冲区封装");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 690), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Buffer 封装 — 统计与 API 对比", nullptr)) {

            if (ImGui::BeginTabBar("BufferTabs")) {

                // ── API 设计 ──────────────────────────────────────────────
                if (ImGui::BeginTabItem("API 设计")) {
                    ImGui::TextColored(ImVec4(1, 0.8f, 0.2f, 1), "Buffer 封装层次（从低到高）");
                    ImGui::Separator();

                    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1, 1), "1. Buffer（基础 RAII）");
                    ImGui::TextWrapped("  Buffer buf;\n"
                                       "  buf.create(dev, {\n"
                                       "      .size    = sizeof(MyVertex) * verts.size(),\n"
                                       "      .usage   = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT\n"
                                       "               | VK_BUFFER_USAGE_TRANSFER_DST_BIT,\n"
                                       "  });\n"
                                       "  buf.upload(dev, verts.data(), buf.size());\n"
                                       "  // buf 析构时自动 vkDestroyBuffer + vkFreeMemory\n");

                    ImGui::Spacing();
                    ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "2. VertexBuffer<T>（类型安全）");
                    ImGui::TextWrapped("  VertexBuffer vb;\n"
                                       "  vb.create(dev, vertices);    // std::vector<T> 直接传入\n"
                                       "  vb.bind(cmd);                // vkCmdBindVertexBuffers\n"
                                       "  vb.draw(cmd);                // vkCmdDraw(cmd, vertexCount, 1, 0, 0)\n");

                    ImGui::Spacing();
                    ImGui::TextColored(ImVec4(1, 0.5f, 0.8f, 1), "3. UniformBuffer<T>（双缓冲 UBO）");
                    ImGui::TextWrapped(
                        "  struct CameraData { mat4 view, proj; vec4 pos; };\n"
                        "  UniformBuffer<CameraData> cameraUBO;\n"
                        "  cameraUBO.create(dev, MAX_FRAMES);   // 每帧一份\n"
                        "  // 每帧：\n"
                        "  cameraUBO.update(fi, { view, proj, camPos });\n"
                        "  auto info = cameraUBO.descriptorInfo(fi);  // 直接用于 vkUpdateDescriptorSets\n");

                    ImGui::Spacing();
                    ImGui::TextColored(ImVec4(1, 0.7f, 0.3f, 1), "4. StagingPool（重用 Staging Buffer）");
                    ImGui::TextWrapped("  StagingPool pool;\n"
                                       "  pool.create(dev, 64 * 1024 * 1024);  // 64 MB 池\n"
                                       "  // 每次上传：\n"
                                       "  auto staging = pool.acquire(dataSize);\n"
                                       "  memcpy(staging.mapped, data, dataSize);\n"
                                       "  // ... vkCmdCopyBuffer ...\n"
                                       "  pool.reset();   // 帧结束归还（不 free，下帧复用）\n");

                    ImGui::EndTabItem();
                }

                // ── 统计面板 ──────────────────────────────────────────────
                if (ImGui::BeginTabItem("运行统计")) {
                    ImGui::TextColored(ImVec4(0.4f, 1, 0.8f, 1), "模拟缓冲区管理器统计");
                    ImGui::Separator();
                    ImGui::Text("总分配次数      : %u", stats_.totalAllocations);
                    ImGui::Text("Staging 分配次数 : %u", stats_.stagingAllocations);
                    ImGui::Text("Staging 缓存命中 : %u", stats_.stagingCacheHits);
                    float hitRate = stats_.stagingAllocations > 0
                                        ? float(stats_.stagingCacheHits) /
                                              float(stats_.stagingAllocations + stats_.stagingCacheHits) * 100.0f
                                        : 0.0f;
                    ImGui::Text("命中率          : %.1f %%", hitRate);
                    ImGui::Spacing();
                    ImGui::Text("总分配大小      : %.1f MB", stats_.totalBytesAllocated / (1024.0f * 1024.0f));
                    ImGui::Text("当前活跃缓冲区  : %u", stats_.activeBuffers);
                    ImGui::Separator();
                    ImGui::Text("本帧 UBO 更新次数 : %u", stats_.uboUpdatesThisFrame);
                    ImGui::Text("场景物体数量     : %zu", simulatedObjects_);
                    ImGui::Spacing();
                    if (ImGui::SliderInt("模拟物体数量", &simulatedObjects_, 1, 500)) {
                        simulateBufferCreation();
                    }
                    ImGui::EndTabItem();
                }

                // ── 对比传统写法 ──────────────────────────────────────────
                if (ImGui::BeginTabItem("代码量对比")) {
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "传统写法（创建 1 个 vertex buffer）");
                    ImGui::Separator();
                    ImGui::TextWrapped("// 手写约 40 行：\n"
                                       "VkBufferCreateInfo bufInfo{};\n"
                                       "bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;\n"
                                       "bufInfo.size  = size;\n"
                                       "bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;\n"
                                       "vkCreateBuffer(device, &bufInfo, nullptr, &stagingBuffer);\n"
                                       "VkMemoryRequirements memReqs;\n"
                                       "vkGetBufferMemoryRequirements(device, stagingBuffer, &memReqs);\n"
                                       "VkMemoryAllocateInfo allocInfo{};\n"
                                       "allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;\n"
                                       "allocInfo.allocationSize = memReqs.size;\n"
                                       "allocInfo.memoryTypeIndex = findMemoryType(...);\n"
                                       "vkAllocateMemory(device, &allocInfo, nullptr, &stagingMem);\n"
                                       "vkBindBufferMemory(device, stagingBuffer, stagingMem, 0);\n"
                                       "void* data;\n"
                                       "vkMapMemory(device, stagingMem, 0, size, 0, &data);\n"
                                       "memcpy(data, vertices.data(), size);\n"
                                       "vkUnmapMemory(device, stagingMem);\n"
                                       "// ... 再写 device-local buffer + copy ... 约再 20 行");

                    ImGui::Spacing();
                    ImGui::TextColored(ImVec4(0.3f, 1, 0.4f, 1), "封装后（1 行）");
                    ImGui::Separator();
                    ImGui::TextWrapped("VertexBuffer vb;\n"
                                       "vb.create(dev, vertices);   // ← 内部自动处理 staging");
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }
        ImGui::End();
    }

  private:
    BufferStats stats_;
    int simulatedObjects_ = 100;
    std::mt19937 rng_;

    void simulateBufferCreation() {
        // 重置统计
        stats_ = {};
        // 顶点缓冲区：每个物体一个 VB
        stats_.totalAllocations += simulatedObjects_;
        stats_.stagingAllocations += simulatedObjects_;   // 首次无缓存
        stats_.stagingCacheHits += simulatedObjects_ * 3; // 后续帧复用
        stats_.totalBytesAllocated += uint64_t(simulatedObjects_) * 6 * sizeof(float) * 36;
        // UBO：2 个（相机 + 光照）+ 每物体 1 个
        stats_.totalAllocations += 2 + simulatedObjects_;
        stats_.totalBytesAllocated += (2 + simulatedObjects_) * 64 * 2; // per-frame * 2
        stats_.activeBuffers = 3 * simulatedObjects_ + 4;               // VB + IB + UBO per obj + shared
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第62章：缓冲区封装（Buffer / VertexBuffer / UniformBuffer）\n";
    std::cout << " 引擎封装系列 — ch62/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch62App app;
        app.run("第62章：缓冲区封装");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
