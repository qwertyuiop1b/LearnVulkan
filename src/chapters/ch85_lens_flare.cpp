/**
 * @file ch85_lens_flare.cpp
 * @brief 第85章：镜头光晕（Lens Flare）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【原理】
 *
 *  强光源（太阳、路灯）进入相机视锥时，镜头内部反射会在传感器上
 *  形成一系列"鬼影"（Ghost）和光晕（Halo）。
 *
 * 【管线】
 *  Bright Pass  → 提取超过阈值的亮像素
 *  Ghost Pass   → 沿光源-屏幕中心轴镜像采样，叠加彩色鬼影
 *  Halo Pass    → 径向模糊形成环形光晕
 *  Streak Pass  → 各向异性模糊形成镜头拉丝
 *  Composite    → 加法混合叠加到 HDR 场景
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <array>
#include <vector>
#include <iostream>

class Ch85App : public DemoApp {
protected:
    void onInit() override { bgColor_ = {0.02f, 0.03f, 0.06f}; }

    void onUpdate() override
    {
        if (animateSun_) {
            sunAngle_ += 0.25f * 0.016f;
            sunScreenPos_.x = 0.5f + 0.35f * std::cos(sunAngle_);
            sunScreenPos_.y = 0.5f + 0.25f * std::sin(sunAngle_);
        }
        recomputeFlare();
    }

    void buildUi() override
    {
        interactive_.buildDebugPanel("第85章：镜头光晕");
        ImGui::Separator();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第85章：镜头光晕（Lens Flare）", nullptr))
        { ImGui::End(); return; }

        if (ImGui::BeginTabBar("FlareTabs")) {
            if (ImGui::BeginTabItem("原理与管线")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "镜头光晕渲染管线");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "1. Bright Pass（亮度提取）\n"
                    "   bright = max(color - threshold, 0)\n\n"
                    "2. Ghost Pass（鬼影）\n"
                    "   ghostUV = screenCenter + (lightUV - center) × ghostScale\n"
                    "   沿光源-中心轴对称采样，叠加 RGB 偏移模拟色散\n\n"
                    "3. Halo Pass（光晕环）\n"
                    "   dist = length(uv - lightUV)\n"
                    "   halo = smoothstep(innerR, outerR, dist) × falloff\n\n"
                    "4. Streak Pass（拉丝）\n"
                    "   沿水平/垂直方向多次采样 bright texture\n\n"
                    "5. Composite\n"
                    "   outColor = sceneColor + flareColor × intensity");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("参数调节")) {
                ImGui::SliderFloat2("光源屏幕位置", &sunScreenPos_.x, 0.0f, 1.0f);
                ImGui::Checkbox("自动移动光源", &animateSun_);
                ImGui::SliderFloat("亮度阈值", &threshold_, 0.5f, 2.5f, "%.2f");
                ImGui::SliderFloat("光晕强度", &intensity_, 0.0f, 3.0f, "%.2f");
                ImGui::SliderInt("鬼影数量", &ghostCount_, 1, 8);
                ImGui::SliderFloat("鬼影间距", &ghostSpacing_, 0.05f, 0.4f, "%.2f");
                ImGui::SliderFloat("光晕半径", &haloRadius_, 0.05f, 0.5f, "%.2f");
                ImGui::SliderFloat("拉丝强度", &streakStrength_, 0.0f, 1.0f, "%.2f");
                ImGui::Spacing();
                ImGui::Text("估算 GPU 开销：%.2f ms（%d Pass）",
                    estimatedGpuMs_, 3 + ghostCount_);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("预览")) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 canvasPos = ImGui::GetCursorScreenPos();
                ImVec2 canvasSize(400, 300);
                dl->AddRectFilled(canvasPos,
                    ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
                    IM_COL32(10, 15, 30, 255));
                ImVec2 center(canvasPos.x + canvasSize.x * 0.5f,
                              canvasPos.y + canvasSize.y * 0.5f);
                ImVec2 lightPos(canvasPos.x + sunScreenPos_.x * canvasSize.x,
                                canvasPos.y + sunScreenPos_.y * canvasSize.y);
                float haloR = haloRadius_ * canvasSize.x;
                dl->AddCircleFilled(lightPos, haloR,
                    IM_COL32(255, 220, 120, int(80 * intensity_)), 32);
                dl->AddCircleFilled(lightPos, 8.0f,
                    IM_COL32(255, 255, 200, 255), 16);
                for (int i = 0; i < ghostCount_; ++i) {
                    float t = (i + 1) * ghostSpacing_;
                    ImVec2 ghost(
                        center.x + (lightPos.x - center.x) * t,
                        center.y + (lightPos.y - center.y) * t);
                    int r = 180 + i * 15;
                    int g = 160 - i * 20;
                    int b = 100 + i * 30;
                    dl->AddCircleFilled(ghost, 6.0f - i * 0.5f,
                        IM_COL32(r, g, b, int(120 * intensity_)), 12);
                }
                if (streakStrength_ > 0.01f) {
                    dl->AddRectFilled(
                        ImVec2(lightPos.x - 80, lightPos.y - 2),
                        ImVec2(lightPos.x + 80, lightPos.y + 2),
                        IM_COL32(255, 240, 180, int(100 * streakStrength_)));
                }
                ImGui::Dummy(canvasSize);
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

private:
    glm::vec2 sunScreenPos_{0.7f, 0.3f};
    float sunAngle_ = 0.0f;
    bool animateSun_ = true;
    float threshold_ = 1.2f;
    float intensity_ = 1.5f;
    int ghostCount_ = 4;
    float ghostSpacing_ = 0.15f;
    float haloRadius_ = 0.18f;
    float streakStrength_ = 0.6f;
    float estimatedGpuMs_ = 0.0f;

    void recomputeFlare()
    {
        float distFromCenter = glm::length(sunScreenPos_ - glm::vec2(0.5f));
        estimatedGpuMs_ = 0.3f + ghostCount_ * 0.08f + streakStrength_ * 0.4f
                        + distFromCenter * 0.2f;
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第85章：镜头光晕（Lens Flare）\n";
    std::cout << " 后处理特效系列 — ch85/6\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch85App app;
        app.run("第85章：镜头光晕");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
