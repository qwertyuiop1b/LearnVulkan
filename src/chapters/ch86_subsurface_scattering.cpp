/**
 * @file ch86_subsurface_scattering.cpp
 * @brief 第86章：次表面散射（Subsurface Scattering / SSS）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【原理】
 *
 *  光线进入半透明材质（皮肤、蜡、玉石、树叶）后在内部散射，
 *  再从其他位置射出，形成柔和的透光效果。
 *
 * 【实现层次】
 *  1. Wrap Lighting — 快速近似，适合实时游戏
 *  2. Separable Blur SSS — 屏幕空间，基于厚度图的可分离模糊
 *  3. Diffusion Profile — 物理准确，预计算 1D 核
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <array>
#include <iostream>

class Ch86App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.06f, 0.04f, 0.05f};
    }

    void onUpdate() override {}

    void buildUi() override {
        interactive_.buildDebugPanel("第86章：次表面散射");
        ImGui::Separator();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第86章：次表面散射（SSS）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("SssTabs")) {
            if (ImGui::BeginTabItem("Wrap Lighting")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Wrap Lighting 近似");
                ImGui::Separator();
                ImGui::TextWrapped("// sss_wrap.frag\n"
                                   "float NdotL = dot(N, L);\n"
                                   "float wrap  = (NdotL + wrapAmount) / (1.0 + wrapAmount);\n"
                                   "wrap = max(wrap, 0.0);\n"
                                   "vec3 sss = albedo * wrap * lightColor;\n\n"
                                   "wrapAmount 越大 → 背光侧也受光 → 更柔和");
                ImGui::SliderFloat("Wrap Amount", &wrapAmount_, 0.0f, 1.0f, "%.2f");
                ImGui::Spacing();
                float ndotlFront = std::max(0.0f, 1.0f);
                float ndotlSide = 0.0f;
                float ndotlBack = -0.5f;
                ImGui::Text("N·L = +1.0（正面）→ wrap = %.3f", computeWrap(ndotlFront));
                ImGui::Text("N·L =  0.0（侧面）→ wrap = %.3f", computeWrap(ndotlSide));
                ImGui::Text("N·L = -0.5（背面）→ wrap = %.3f", computeWrap(ndotlBack));
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Separable Blur")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "屏幕空间可分离模糊 SSS");
                ImGui::Separator();
                ImGui::TextWrapped("Pass 1: 渲染 Diffuse + Thickness（厚度图）\n"
                                   "Pass 2: 水平模糊 — 采样半径由 thickness 控制\n"
                                   "Pass 3: 垂直模糊\n"
                                   "Pass 4: 与原始 Diffuse 混合\n\n"
                                   "  blurRadius = baseRadius × thickness × scatterScale");
                ImGui::SliderFloat("散射强度", &scatterScale_, 0.1f, 4.0f, "%.2f");
                ImGui::SliderInt("模糊采样数", &blurSamples_, 4, 32);
                ImGui::SliderFloat("厚度乘子", &thicknessMult_, 0.1f, 2.0f, "%.2f");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("材质预设")) {
                const char* presets[] = {"皮肤", "蜡", "玉石", "树叶", "牛奶"};
                if (ImGui::Combo("材质类型", &materialPreset_, presets, 5))
                    applyPreset(materialPreset_);
                ImGui::Spacing();
                ImGui::ColorEdit3("散射颜色", &scatterColor_.x);
                ImGui::SliderFloat("散射距离 (mm)", &scatterDist_, 0.1f, 25.0f, "%.1f");
                ImGui::SliderFloat("透射强度", &transmission_, 0.0f, 1.0f, "%.2f");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "材质参数预览：");
                ImGui::Text("  散射 RGB : (%.2f, %.2f, %.2f)", scatterColor_.x, scatterColor_.y, scatterColor_.z);
                ImGui::Text("  估算模糊半径 : %.1f px", scatterDist_ * scatterScale_ * 2.0f);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Diffusion Profile")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "1D 扩散剖面（物理准确）");
                ImGui::Separator();
                ImGui::TextWrapped("预计算 RGB 各自的 1D 扩散核（存储为 1D Texture）：\n"
                                   "  · 皮肤：R 散射远，G 中等，B 近\n"
                                   "  · 采样：integrate(diffusionProfile, distance)\n\n"
                                   "Jimenez Separable SSS（2015）：\n"
                                   "  将 2D 核分解为 X/Y 两个 1D 核 → 两次 Pass");
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                float w = 400.0f;
                float h = 120.0f;
                for (int ch = 0; ch < 3; ++ch) {
                    float yOff = ch * (h / 3.0f + 4.0f);
                    ImU32 col = ch == 0   ? IM_COL32(255, 80, 80, 255)
                                : ch == 1 ? IM_COL32(80, 255, 80, 255)
                                          : IM_COL32(80, 80, 255, 255);
                    for (int i = 0; i < 100; ++i) {
                        float t = float(i) / 100.0f;
                        float profile = std::exp(-t * (3.0f + ch * 2.0f));
                        float px = pos.x + t * w;
                        float py = pos.y + yOff + h / 3.0f - profile * (h / 3.0f - 4);
                        dl->AddCircleFilled(ImVec2(px, py), 2.0f, col);
                    }
                }
                ImGui::Dummy(ImVec2(w, h + 12));
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    float wrapAmount_ = 0.5f;
    float scatterScale_ = 1.5f;
    int blurSamples_ = 16;
    float thicknessMult_ = 1.0f;
    int materialPreset_ = 0;
    glm::vec3 scatterColor_{1.0f, 0.7f, 0.6f};
    float scatterDist_ = 3.0f;
    float transmission_ = 0.4f;

    float computeWrap(float ndotl) const {
        return std::max(0.0f, (ndotl + wrapAmount_) / (1.0f + wrapAmount_));
    }

    void applyPreset(int preset) {
        switch (preset) {
        case 0:
            scatterColor_ = {1.0f, 0.7f, 0.6f};
            scatterDist_ = 3.0f;
            wrapAmount_ = 0.5f;
            break;
        case 1:
            scatterColor_ = {1.0f, 0.95f, 0.8f};
            scatterDist_ = 8.0f;
            wrapAmount_ = 0.7f;
            break;
        case 2:
            scatterColor_ = {0.7f, 0.9f, 0.7f};
            scatterDist_ = 2.0f;
            wrapAmount_ = 0.3f;
            break;
        case 3:
            scatterColor_ = {0.5f, 0.8f, 0.3f};
            scatterDist_ = 1.5f;
            wrapAmount_ = 0.6f;
            break;
        case 4:
            scatterColor_ = {0.95f, 0.95f, 0.9f};
            scatterDist_ = 12.0f;
            wrapAmount_ = 0.8f;
            break;
        default:
            break;
        }
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第86章：次表面散射（SSS）\n";
    std::cout << " 材质渲染系列 — ch86/6\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch86App app;
        app.run("第86章：次表面散射");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
