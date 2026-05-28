/**
 * @file ch93_procedural_animation.cpp
 * @brief 第93章：程序化动画（IK / 状态机 / Blend Tree）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【三层动画系统】
 *
 *  AnimationStateMachine — Idle / Walk / Run / Jump 状态切换
 *  BlendTree             — 1D/2D 混合（速度 → 动画权重）
 *  TwoBoneIK             — 手臂/腿部 IK 求解（FABRIK 简化版）
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <array>
#include <vector>
#include <iostream>

enum class AnimState { Idle, Walk, Run, Jump, Attack };

struct Bone2D {
    glm::vec2 pos{0.0f};
    float angle = 0.0f;
    float length = 0.0f;
};

class Ch93App : public DemoApp {
protected:
    void onInit() override
    {
        bgColor_ = {0.06f, 0.05f, 0.08f};
        setupSkeleton();
    }

    void onUpdate() override
    {
        elapsed_ += 0.016f;
        updateStateMachine();
        updateBlendTree();
        solveTwoBoneIK();
        animateSkeleton();
    }

    void buildUi() override
    {
        interactive_.buildDebugPanel("第93章：程序化动画");
        ImGui::Separator();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第93章：程序化动画（IK + Blend Tree）", nullptr))
        { ImGui::End(); return; }

        if (ImGui::BeginTabBar("AnimTabs")) {
            if (ImGui::BeginTabItem("状态机")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Animation State Machine");
                ImGui::Separator();
                const char* states[] = {"Idle", "Walk", "Run", "Jump", "Attack"};
                int s = static_cast<int>(currentState_);
                ImGui::Text("当前状态 : %s", states[s]);
                ImGui::SliderFloat("移动速度", &moveSpeed_, 0.0f, 8.0f, "%.1f m/s");
                ImGui::Checkbox("在地面上", &onGround_);
                ImGui::Checkbox("攻击键", &attackInput_);
                ImGui::Spacing();
                ImGui::TextWrapped(
                    "转换规则：\n"
                    "  Idle  → Walk   : speed > 0.1\n"
                    "  Walk  → Run    : speed > 4.0\n"
                    "  *     → Jump   : !onGround\n"
                    "  *     → Attack : attackInput（优先级最高）\n"
                    "  任意  → Idle   : speed < 0.1 && onGround");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Blend Tree")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "1D Blend Tree（速度驱动）");
                ImGui::Separator();
                ImGui::Text("Idle 权重 : %.2f", blendWeights_[0]);
                ImGui::ProgressBar(blendWeights_[0], ImVec2(-1, 14));
                ImGui::Text("Walk 权重 : %.2f", blendWeights_[1]);
                ImGui::ProgressBar(blendWeights_[1], ImVec2(-1, 14));
                ImGui::Text("Run  权重 : %.2f", blendWeights_[2]);
                ImGui::ProgressBar(blendWeights_[2], ImVec2(-1, 14));
                ImGui::Spacing();
                ImGui::TextWrapped(
                    "// 1D Blend Tree\n"
                    "float wWalk = saturate(speed / walkThreshold);\n"
                    "float wRun  = saturate((speed - walkThreshold) / (runThreshold - walkThreshold));\n"
                    "float wIdle = 1.0 - wWalk;\n"
                    "pose = wIdle*poseIdle + wWalk*poseWalk + wRun*poseRun;");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Two-Bone IK")) {
                ImGui::SliderFloat2("IK 目标", &ikTarget_.x, -1.5f, 1.5f);
                ImGui::Checkbox("启用 IK", &enableIk_);
                ImGui::Spacing();
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 origin = ImGui::GetCursorScreenPos();
                origin.x += 220.0f;
                origin.y += 180.0f;
                float scale = 80.0f;
                auto drawBone = [&](const Bone2D& a, const Bone2D& b, ImU32 col) {
                    ImVec2 p0(origin.x + a.pos.x * scale, origin.y - a.pos.y * scale);
                    ImVec2 p1(origin.x + b.pos.x * scale, origin.y - b.pos.y * scale);
                    dl->AddLine(p0, p1, col, 4.0f);
                    dl->AddCircleFilled(p0, 5.0f, col);
                    dl->AddCircleFilled(p1, 5.0f, col);
                };
                drawBone(root_, upperArm_, IM_COL32(200, 180, 120, 255));
                drawBone(upperArm_, lowerArm_, IM_COL32(200, 180, 120, 255));
                ImVec2 targetScreen(origin.x + ikTarget_.x * scale,
                                    origin.y - ikTarget_.y * scale);
                dl->AddCircle(targetScreen, 6.0f, IM_COL32(255, 80, 80, 255), 12, 2.0f);
                ImGui::Dummy(ImVec2(440, 280));
                ImGui::Text("上臂角 : %.1f°  前臂角 : %.1f°",
                    glm::degrees(upperArm_.angle), glm::degrees(lowerArm_.angle));
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("完整骨架")) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 origin = ImGui::GetCursorScreenPos();
                origin.x += 220.0f;
                origin.y += 20.0f;
                float sc = 60.0f;
                for (size_t i = 0; i + 1 < skeleton_.size(); ++i) {
                    ImVec2 p0(origin.x + skeleton_[i].pos.x * sc,
                              origin.y + skeleton_[i].pos.y * sc);
                    ImVec2 p1(origin.x + skeleton_[i + 1].pos.x * sc,
                              origin.y + skeleton_[i + 1].pos.y * sc);
                    dl->AddLine(p0, p1, IM_COL32(100, 200, 255, 255), 3.0f);
                    dl->AddCircleFilled(p0, 4.0f, IM_COL32(255, 220, 100, 255));
                }
                ImGui::Dummy(ImVec2(440, 320));
                ImGui::Text("动画时间 : %.2f s  |  状态 : %d", elapsed_, static_cast<int>(currentState_));
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

private:
    AnimState currentState_ = AnimState::Idle;
    float moveSpeed_ = 0.0f;
    bool onGround_ = true;
    bool attackInput_ = false;
    bool enableIk_ = true;
    float elapsed_ = 0.0f;
    std::array<float, 3> blendWeights_{1.0f, 0.0f, 0.0f};
    glm::vec2 ikTarget_{0.8f, 0.5f};
    Bone2D root_{{0.0f, 0.0f}, 0.0f, 0.0f};
    Bone2D upperArm_{{0.0f, 0.0f}, 0.0f, 0.5f};
    Bone2D lowerArm_{{0.5f, 0.0f}, 0.0f, 0.4f};
    std::vector<Bone2D> skeleton_;

    void setupSkeleton()
    {
        skeleton_ = {
            {{0.0f, 0.0f}, 0.0f, 0.0f},
            {{0.0f, 0.8f}, 0.0f, 0.0f},
            {{0.0f, 1.5f}, 0.0f, 0.0f},
            {{-0.3f, 1.3f}, 0.0f, 0.0f},
            {{-0.5f, 1.0f}, 0.0f, 0.0f},
            {{0.3f, 1.3f}, 0.0f, 0.0f},
            {{0.5f, 1.0f}, 0.0f, 0.0f},
            {{-0.15f, 0.0f}, 0.0f, 0.0f},
            {{-0.15f, -0.7f}, 0.0f, 0.0f},
            {{0.15f, 0.0f}, 0.0f, 0.0f},
            {{0.15f, -0.7f}, 0.0f, 0.0f},
        };
        upperArm_.pos = {0.0f, 0.0f};
        upperArm_.length = 0.5f;
        lowerArm_.pos = {0.5f, 0.0f};
        lowerArm_.length = 0.4f;
    }

    void updateStateMachine()
    {
        moveSpeed_ = 2.0f + 2.0f * std::abs(std::sin(elapsed_ * 0.5f));
        if (attackInput_) {
            currentState_ = AnimState::Attack;
            return;
        }
        if (!onGround_) {
            currentState_ = AnimState::Jump;
            return;
        }
        if (moveSpeed_ > 4.0f) currentState_ = AnimState::Run;
        else if (moveSpeed_ > 0.1f) currentState_ = AnimState::Walk;
        else currentState_ = AnimState::Idle;
    }

    void updateBlendTree()
    {
        const float walkThreshold = 0.1f;
        const float runThreshold = 4.0f;
        float wWalk = std::clamp(moveSpeed_ / 2.0f, 0.0f, 1.0f);
        float wRun  = std::clamp((moveSpeed_ - walkThreshold) / (runThreshold - walkThreshold), 0.0f, 1.0f);
        blendWeights_[2] = wRun;
        blendWeights_[1] = wWalk * (1.0f - wRun);
        blendWeights_[0] = 1.0f - blendWeights_[1] - blendWeights_[2];
    }

    void solveTwoBoneIK()
    {
        if (!enableIk_) return;
        float L1 = upperArm_.length;
        float L2 = lowerArm_.length;
        glm::vec2 target = ikTarget_;
        float dist = glm::length(target);
        dist = std::clamp(dist, 0.01f, L1 + L2 - 0.01f);
        float cosAngle2 = (dist * dist - L1 * L1 - L2 * L2) / (2.0f * L1 * L2);
        cosAngle2 = std::clamp(cosAngle2, -1.0f, 1.0f);
        float angle2 = std::acos(cosAngle2);
        float angle1 = std::atan2(target.y, target.x) - std::atan2(L2 * std::sin(angle2), L1 + L2 * std::cos(angle2));
        upperArm_.angle = angle1;
        lowerArm_.angle = angle1 + angle2;
        upperArm_.pos = {0.0f, 0.0f};
        lowerArm_.pos = {
            std::cos(angle1) * L1,
            std::sin(angle1) * L1
        };
    }

    void animateSkeleton()
    {
        float swing = std::sin(elapsed_ * moveSpeed_) * 0.15f * blendWeights_[1];
        float runSwing = std::sin(elapsed_ * moveSpeed_ * 2.0f) * 0.25f * blendWeights_[2];
        skeleton_[3].pos.x = -0.3f + swing + runSwing;
        skeleton_[5].pos.x = 0.3f - swing - runSwing;
        skeleton_[8].pos.y = -0.7f + std::abs(std::sin(elapsed_ * moveSpeed_)) * 0.1f;
        skeleton_[10].pos.y = -0.7f + std::abs(std::cos(elapsed_ * moveSpeed_)) * 0.1f;
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第93章：程序化动画\n";
    std::cout << " 游戏引擎系列 — ch93/4\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch93App app;
        app.run("第93章：程序化动画");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
