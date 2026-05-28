/**
 * @file ch92_network_sync.cpp
 * @brief 第92章：网络同步（Client Prediction + State Interpolation）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【同步模型】
 *
 *  Server Authoritative — 服务器权威，客户端预测
 *  Client Prediction    — 本地立即响应输入，等待服务器校正
 *  Entity Interpolation — 其他玩家位置在两帧快照间插值
 *  Snapshot Buffer      — 延迟渲染以平滑远程实体
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <array>
#include <vector>
#include <deque>
#include <iostream>

struct NetSnapshot {
    float serverTime = 0.0f;
    glm::vec2 playerPos{0.0f};
    glm::vec2 remotePos{0.0f};
    int inputSeq = 0;
};

struct PendingInput {
    int seq = 0;
    glm::vec2 moveDir{0.0f};
    float timestamp = 0.0f;
};

class Ch92App : public DemoApp {
protected:
    void onInit() override { bgColor_ = {0.04f, 0.05f, 0.08f}; }

    void onUpdate() override
    {
        elapsed_ += 0.016f;
        simulateNetwork();
        if (enablePrediction_) applyClientPrediction();
        updateInterpolation();
    }

    void buildUi() override
    {
        interactive_.buildDebugPanel("第92章：网络同步");
        ImGui::Separator();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第92章：网络同步（Prediction + Interpolation）", nullptr))
        { ImGui::End(); return; }

        if (ImGui::BeginTabBar("NetTabs")) {
            if (ImGui::BeginTabItem("同步模型")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Client-Server 同步");
                ImGui::Separator();
                ImGui::Checkbox("客户端预测", &enablePrediction_);
                ImGui::Checkbox("实体插值", &enableInterpolation_);
                ImGui::SliderInt("模拟 RTT (ms)", &simulatedRttMs_, 20, 300);
                ImGui::SliderFloat("插值延迟 (s)", &interpDelay_, 0.05f, 0.3f, "%.2f");
                ImGui::Spacing();
                ImGui::TextWrapped(
                    "客户端预测流程：\n"
                    "  1. 收到输入 → 立即本地移动（predictedPos）\n"
                    "  2. 发送 InputCmd(seq, dir) 到服务器\n"
                    "  3. 收到 Snapshot → 若偏差 > 阈值，回滚并重放\n\n"
                    "实体插值（远程玩家）：\n"
                    "  renderTime = serverTime - interpDelay\n"
                    "  pos = lerp(snapshot[t0], snapshot[t1], alpha)");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("可视化")) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                ImVec2 size(440, 280);
                dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                    IM_COL32(20, 25, 35, 255));
                auto toScreen = [&](glm::vec2 p) {
                    return ImVec2(pos.x + (p.x + 1.0f) * 0.5f * size.x,
                                  pos.y + (1.0f - (p.y + 1.0f) * 0.5f) * size.y);
                };
                dl->AddCircleFilled(toScreen(serverPos_), 8.0f, IM_COL32(255, 80, 80, 255));
                dl->AddText(toScreen(serverPos_ + glm::vec2(0.05f, 0.05f)), IM_COL32(255, 150, 150, 255), "Server");
                dl->AddCircleFilled(toScreen(predictedPos_), 7.0f, IM_COL32(80, 255, 120, 255));
                dl->AddText(toScreen(predictedPos_ + glm::vec2(0.05f, -0.08f)), IM_COL32(150, 255, 150, 255), "Predicted");
                dl->AddCircleFilled(toScreen(interpolatedRemote_), 7.0f, IM_COL32(100, 180, 255, 255));
                dl->AddText(toScreen(interpolatedRemote_ + glm::vec2(0.05f, 0.05f)), IM_COL32(150, 200, 255, 255), "Remote");
                if (correctionMag_ > 0.01f) {
                    ImVec2 from = toScreen(predictedPos_);
                    ImVec2 to   = toScreen(serverPos_);
                    dl->AddLine(from, to, IM_COL32(255, 255, 0, 200), 2.0f);
                }
                ImGui::Dummy(size);
                ImGui::Text("预测误差 : %.4f  |  校正次数 : %d", correctionMag_, correctionCount_);
                ImGui::Text("快照缓冲 : %zu  |  待确认输入 : %zu", snapshots_.size(), pendingInputs_.size());
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("协议设计")) {
                ImGui::TextWrapped(
                    "// 客户端 → 服务器\n"
                    "struct InputCmd {\n"
                    "    uint32_t seq;\n"
                    "    float    dt;\n"
                    "    vec2     moveDir;\n"
                    "    uint32_t buttons;\n"
                    "};\n\n"
                    "// 服务器 → 客户端\n"
                    "struct WorldSnapshot {\n"
                    "    float    serverTime;\n"
                    "    uint32_t ackInputSeq;  // 确认到的输入序号\n"
                    "    EntityState entities[MAX_ENTITIES];\n"
                    "};\n\n"
                    "回滚重放：\n"
                    "  savedState = serverSnapshot.state[localPlayer]\n"
                    "  for (input : pendingInputs where input.seq > ackSeq)\n"
                    "      simulate(savedState, input)\n"
                    "  predictedPos = savedState.position");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("带宽优化")) {
                ImGui::TextWrapped(
                    "Delta Compression — 只发送相对上一帧的变化\n"
                    "Interest Management — 只同步视野内实体\n"
                    "Quantization — 位置 float32 → int16（精度 1cm）\n"
                    "  posX = int16(worldX * 100)\n\n"
                    "UDP + 可靠层（ENet / GameNetworkingSockets）\n"
                    "  · InputCmd：可靠有序\n"
                    "  · Snapshot：不可靠最新优先");
                ImGui::Spacing();
                ImGui::Text("估算带宽 : %.1f KB/s（%d 玩家）", bandwidthKbps_ / 8.0f, playerCount_);
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

private:
    glm::vec2 serverPos_{0.0f, 0.0f};
    glm::vec2 predictedPos_{0.0f, 0.0f};
    glm::vec2 interpolatedRemote_{0.5f, 0.3f};
    glm::vec2 remoteServerPos_{0.5f, 0.3f};
    float elapsed_ = 0.0f;
    bool enablePrediction_ = true;
    bool enableInterpolation_ = true;
    int simulatedRttMs_ = 80;
    float interpDelay_ = 0.1f;
    float correctionMag_ = 0.0f;
    int correctionCount_ = 0;
    int inputSeq_ = 0;
    int playerCount_ = 8;
    float bandwidthKbps_ = 12.0f;
    std::deque<NetSnapshot> snapshots_;
    std::deque<PendingInput> pendingInputs_;
    float snapshotTimer_ = 0.0f;

    void simulateNetwork()
    {
        float moveSpeed = 0.8f;
        glm::vec2 inputDir{std::cos(elapsed_ * 0.7f), std::sin(elapsed_ * 0.5f)};
        inputDir = glm::normalize(inputDir) * 0.016f * moveSpeed;
        serverPos_ += inputDir;
        serverPos_ = glm::clamp(serverPos_, glm::vec2(-0.9f), glm::vec2(0.9f));
        remoteServerPos_.x = 0.4f * std::cos(elapsed_ * 0.4f);
        remoteServerPos_.y = 0.4f * std::sin(elapsed_ * 0.6f);
        ++inputSeq_;
        pendingInputs_.push_back({inputSeq_, inputDir, elapsed_});
        if (pendingInputs_.size() > 30) pendingInputs_.pop_front();
        snapshotTimer_ += 0.016f;
        float snapInterval = float(simulatedRttMs_) / 1000.0f * 0.5f;
        if (snapshotTimer_ >= snapInterval) {
            snapshotTimer_ = 0.0f;
            snapshots_.push_back({elapsed_, serverPos_, remoteServerPos_, inputSeq_});
            if (snapshots_.size() > 20) snapshots_.pop_front();
            if (enablePrediction_ && !snapshots_.empty()) {
                glm::vec2 diff = predictedPos_ - serverPos_;
                correctionMag_ = glm::length(diff);
                if (correctionMag_ > 0.05f) {
                    ++correctionCount_;
                    predictedPos_ = glm::mix(predictedPos_, serverPos_, 0.3f);
                }
            }
        }
        bandwidthKbps_ = 8.0f + playerCount_ * 1.5f + simulatedRttMs_ * 0.02f;
    }

    void applyClientPrediction()
    {
        if (pendingInputs_.empty()) return;
        predictedPos_ += pendingInputs_.back().moveDir;
        predictedPos_ = glm::clamp(predictedPos_, glm::vec2(-0.9f), glm::vec2(0.9f));
    }

    void updateInterpolation()
    {
        if (!enableInterpolation_ || snapshots_.size() < 2) {
            interpolatedRemote_ = remoteServerPos_;
            return;
        }
        float renderTime = elapsed_ - interpDelay_;
        size_t i1 = snapshots_.size() - 1;
        size_t i0 = i1 > 0 ? i1 - 1 : 0;
        float t0 = snapshots_[i0].serverTime;
        float t1 = snapshots_[i1].serverTime;
        float alpha = t1 > t0 ? (renderTime - t0) / (t1 - t0) : 0.0f;
        alpha = std::clamp(alpha, 0.0f, 1.0f);
        interpolatedRemote_ = glm::mix(snapshots_[i0].remotePos, snapshots_[i1].remotePos, alpha);
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第92章：网络同步\n";
    std::cout << " 游戏引擎系列 — ch92/4\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch92App app;
        app.run("第92章：网络同步");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
