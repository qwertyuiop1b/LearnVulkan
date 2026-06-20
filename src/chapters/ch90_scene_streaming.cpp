/**
 * @file ch90_scene_streaming.cpp
 * @brief 第90章：场景流式加载（Scene Streaming）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【核心概念】
 *
 *  开放世界无法一次性加载全部资源，需要按玩家位置
 *  动态加载/卸载场景区块（Chunk）。
 *
 * 【架构】
 *  WorldGrid        — 将世界划分为 Chunk（如 256m × 256m）
 *  StreamingRing    — 以玩家为中心的多环加载区
 *  AsyncLoadQueue   — 后台线程加载，主线程提交 GPU 资源
 *  MemoryBudget     — 超出预算时 LRU 卸载最远 Chunk
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <array>
#include <vector>
#include <deque>
#include <algorithm>
#include <iostream>

enum class ChunkState { Unloaded, Loading, Loaded, Unloading };

struct ChunkInfo {
    int cx, cz;
    ChunkState state = ChunkState::Unloaded;
    float loadProgress = 0.0f;
    size_t memoryBytes = 0;
    float lastAccessTime = 0.0f;
};

class Ch90App : public DemoApp {
  protected:
    static constexpr int WORLD_CHUNKS = 16;
    static constexpr float CHUNK_SIZE = 256.0f;

    void onInit() override {
        bgColor_ = {0.04f, 0.05f, 0.07f};
    }

    void onUpdate() override {
        elapsed_ += 0.016f;
        if (simulatePlayer_) {
            playerPos_.x = std::cos(elapsed_ * 0.3f) * CHUNK_SIZE * 3.0f;
            playerPos_.z = std::sin(elapsed_ * 0.2f) * CHUNK_SIZE * 3.0f;
        }
        updateStreaming();
        processLoadQueue();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第90章：场景流式加载");
        ImGui::Separator();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第90章：场景流式加载（Scene Streaming）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("StreamTabs")) {
            if (ImGui::BeginTabItem("世界网格")) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                float cs = 20.0f;
                ImVec2 gridSize(WORLD_CHUNKS * cs, WORLD_CHUNKS * cs);
                int pcx = chunkCoord(playerPos_.x);
                int pcz = chunkCoord(playerPos_.z);
                for (int cz = 0; cz < WORLD_CHUNKS; ++cz) {
                    for (int cx = 0; cx < WORLD_CHUNKS; ++cx) {
                        int idx = cz * WORLD_CHUNKS + cx;
                        ImVec2 c0(pos.x + cx * cs, pos.y + cz * cs);
                        ImVec2 c1(c0.x + cs, c0.y + cs);
                        ImU32 col = chunkColor(chunks_[idx].state);
                        dl->AddRectFilled(c0, c1, col);
                        int dist = std::max(std::abs(cx - pcx), std::abs(cz - pcz));
                        if (dist <= loadRadius_)
                            dl->AddRect(c0, c1, IM_COL32(255, 255, 100, 200), 0, 0, 2.0f);
                        if (dist <= unloadRadius_)
                            dl->AddRect(c0, c1, IM_COL32(100, 255, 100, 100), 0, 0, 1.0f);
                    }
                }
                float px = pos.x + (pcx + 0.5f) * cs;
                float pz = pos.y + (pcz + 0.5f) * cs;
                dl->AddCircleFilled(ImVec2(px, pz), 4.0f, IM_COL32(255, 80, 80, 255));
                ImGui::Dummy(gridSize);
                ImGui::Text("图例：灰=未加载  黄=加载中  绿=已加载  红=卸载中");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("流式参数")) {
                ImGui::SliderInt("加载半径 (Chunk)", &loadRadius_, 1, 6);
                ImGui::SliderInt("卸载半径 (Chunk)", &unloadRadius_, 2, 8);
                if (unloadRadius_ <= loadRadius_)
                    unloadRadius_ = loadRadius_ + 1;
                ImGui::SliderFloat("内存预算 (MB)", &memoryBudgetMB_, 64.0f, 512.0f, "%.0f");
                ImGui::SliderInt("每帧最大加载数", &maxLoadsPerFrame_, 1, 4);
                ImGui::Checkbox("模拟玩家移动", &simulatePlayer_);
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "当前状态：");
                ImGui::Text("  已加载 Chunk : %d / %d", loadedCount_, WORLD_CHUNKS * WORLD_CHUNKS);
                ImGui::Text("  内存使用     : %.1f / %.0f MB", usedMemoryMB_, memoryBudgetMB_);
                ImGui::Text("  加载队列     : %zu 个", loadQueue_.size());
                ImGui::ProgressBar(usedMemoryMB_ / memoryBudgetMB_, ImVec2(-1, 18));
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("AsyncLoadQueue")) {
                ImGui::TextWrapped("class AsyncLoadQueue {\n"
                                   "    std::thread              worker_;\n"
                                   "    std::mutex               mutex_;\n"
                                   "    std::deque<LoadRequest>  pending_;\n"
                                   "    std::deque<LoadResult>   completed_;\n\n"
                                   "    // 工作线程：磁盘 IO + 反序列化\n"
                                   "    void workerLoop() {\n"
                                   "        while (running_) {\n"
                                   "            auto req = popPending();\n"
                                   "            ChunkData data = loadFromDisk(req.path);\n"
                                   "            pushCompleted({req.id, std::move(data)});\n"
                                   "        }\n"
                                   "    }\n\n"
                                   "    // 主线程：每帧 poll completed，创建 VkBuffer/VkImage\n"
                                   "    void uploadToGpu(LoadResult& r) {\n"
                                   "        stagingPool_.upload(r.mesh.vertices, vb_);\n"
                                   "        stagingPool_.upload(r.mesh.indices,  ib_);\n"
                                   "    }\n"
                                   "};");
                ImGui::Spacing();
                ImGui::Text("本帧处理加载 : %d  本帧卸载 : %d", loadsThisFrame_, unloadsThisFrame_);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("LRU 卸载")) {
                ImGui::TextWrapped("当内存超出 budget 时，按 LRU 卸载最久未访问的 Chunk：\n\n"
                                   "  sort(chunks, [](a,b){ return a.lastAccess < b.lastAccess; })\n"
                                   "  while (usedMemory > budget) unload(chunks.front())\n\n"
                                   "  lastAccessTime 在玩家进入 Chunk 或引用资源时更新");
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    std::array<ChunkInfo, WORLD_CHUNKS * WORLD_CHUNKS> chunks_{};
    glm::vec3 playerPos_{0.0f, 0.0f, 0.0f};
    float elapsed_ = 0.0f;
    bool simulatePlayer_ = true;
    int loadRadius_ = 3;
    int unloadRadius_ = 5;
    float memoryBudgetMB_ = 256.0f;
    float usedMemoryMB_ = 0.0f;
    int maxLoadsPerFrame_ = 2;
    int loadedCount_ = 0;
    int loadsThisFrame_ = 0;
    int unloadsThisFrame_ = 0;
    std::deque<int> loadQueue_;

    int chunkCoord(float worldPos) const {
        int c = int(std::floor(worldPos / CHUNK_SIZE)) + WORLD_CHUNKS / 2;
        return std::clamp(c, 0, WORLD_CHUNKS - 1);
    }

    static ImU32 chunkColor(ChunkState s) {
        switch (s) {
        case ChunkState::Unloaded:
            return IM_COL32(50, 50, 55, 255);
        case ChunkState::Loading:
            return IM_COL32(200, 180, 50, 255);
        case ChunkState::Loaded:
            return IM_COL32(50, 140, 70, 255);
        case ChunkState::Unloading:
            return IM_COL32(180, 50, 50, 255);
        }
        return IM_COL32(80, 80, 80, 255);
    }

    void updateStreaming() {
        int pcx = chunkCoord(playerPos_.x);
        int pcz = chunkCoord(playerPos_.z);
        loadsThisFrame_ = 0;
        unloadsThisFrame_ = 0;
        loadedCount_ = 0;
        usedMemoryMB_ = 0.0f;
        for (int cz = 0; cz < WORLD_CHUNKS; ++cz) {
            for (int cx = 0; cx < WORLD_CHUNKS; ++cx) {
                int idx = cz * WORLD_CHUNKS + cx;
                int dist = std::max(std::abs(cx - pcx), std::abs(cz - pcz));
                auto& ch = chunks_[idx];
                ch.cx = cx;
                ch.cz = cz;
                if (dist <= loadRadius_ && ch.state == ChunkState::Unloaded) {
                    ch.state = ChunkState::Loading;
                    ch.loadProgress = 0.0f;
                    loadQueue_.push_back(idx);
                }
                if (dist > unloadRadius_ && ch.state == ChunkState::Loaded) {
                    ch.state = ChunkState::Unloading;
                    ++unloadsThisFrame_;
                }
                if (ch.state == ChunkState::Loaded) {
                    ++loadedCount_;
                    usedMemoryMB_ += ch.memoryBytes / (1024.0f * 1024.0f);
                    ch.lastAccessTime = elapsed_;
                }
                if (ch.state == ChunkState::Unloading) {
                    ch.loadProgress -= 0.05f;
                    if (ch.loadProgress <= 0.0f) {
                        ch.state = ChunkState::Unloaded;
                        ch.memoryBytes = 0;
                    }
                }
            }
        }
        while (usedMemoryMB_ > memoryBudgetMB_) {
            int oldest = -1;
            float oldestTime = 1e9f;
            for (int i = 0; i < WORLD_CHUNKS * WORLD_CHUNKS; ++i) {
                if (chunks_[i].state == ChunkState::Loaded && chunks_[i].lastAccessTime < oldestTime) {
                    oldestTime = chunks_[i].lastAccessTime;
                    oldest = i;
                }
            }
            if (oldest < 0)
                break;
            chunks_[oldest].state = ChunkState::Unloading;
            chunks_[oldest].loadProgress = 1.0f;
            ++unloadsThisFrame_;
            usedMemoryMB_ -= chunks_[oldest].memoryBytes / (1024.0f * 1024.0f);
        }
    }

    void processLoadQueue() {
        int processed = 0;
        while (!loadQueue_.empty() && processed < maxLoadsPerFrame_) {
            int idx = loadQueue_.front();
            loadQueue_.pop_front();
            auto& ch = chunks_[idx];
            if (ch.state != ChunkState::Loading)
                continue;
            ch.loadProgress += 0.15f;
            if (ch.loadProgress >= 1.0f) {
                ch.state = ChunkState::Loaded;
                ch.memoryBytes = 4 * 1024 * 1024;
                ++loadsThisFrame_;
            } else {
                loadQueue_.push_back(idx);
            }
            ++processed;
        }
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第90章：场景流式加载\n";
    std::cout << " 引擎架构系列 — ch90/6\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch90App app;
        app.run("第90章：场景流式加载");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
