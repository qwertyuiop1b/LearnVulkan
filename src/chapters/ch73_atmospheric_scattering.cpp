/**
 * @file ch73_atmospheric_scattering.cpp
 * @brief 第73章：大气散射（Rayleigh + Mie Scattering）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【物理原理】
 *
 *  Rayleigh 散射（瑞利散射）：
 *    · 散射体尺寸 << 波长（氮气、氧气分子）
 *    · 散射强度 ∝ λ^-4（波长越短散射越强）
 *    · 蓝光(450nm) >> 红光(700nm) → 天空是蓝色
 *    · 散射方向分布：1 + cos²(θ)（前后对称）
 *
 *  Mie 散射（米氏散射）：
 *    · 散射体尺寸 ≈ 波长（灰尘、气溶胶、水雾）
 *    · 对各波长散射程度相近 → 白色光晕
 *    · 方向性强（Henyey-Greenstein 相位函数），主要向前散射
 *    · g 因子（不对称参数）：0=均匀, 0.9=强烈前向散射
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <iostream>

static constexpr float PI = 3.14159265f;

class Ch73App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.05f, 0.10f, 0.18f};
        updateSkyColor();
    }

    void onUpdate() override {
        updateSkyColor();
        bgColor_ = {zenithColor_.x * 0.5f, zenithColor_.y * 0.5f, zenithColor_.z * 0.5f};
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第73章：大气散射");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第73章：大气散射（Rayleigh + Mie）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("AtmoTabs")) {

            // ── Tab 1: 物理原理 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("物理原理")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "大气散射物理基础");
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.4f, 0.7f, 1, 1), "Rayleigh 散射（小粒子，分子级）");
                ImGui::TextWrapped("散射强度：β_R(λ) = 8π³(n²-1)² / (3Nλ⁴)\n\n"
                                   "  · λ = 波长（m）：450nm(蓝) / 550nm(绿) / 700nm(红)\n"
                                   "  · n = 折射率（空气 ≈ 1.0003）\n"
                                   "  · N = 单位体积分子数\n\n"
                                   "  相位函数（散射方向分布）：\n"
                                   "    P_R(θ) = (3/16π)(1 + cos²θ)\n"
                                   "  → 前向和后向散射相等（太阳方向和反方向都会散射）\n\n"
                                   "  为什么天空是蓝色：\n"
                                   "    β_R(450nm) / β_R(700nm) = (700/450)^4 ≈ 5.5\n"
                                   "  → 蓝光散射强度是红光的 5.5 倍！");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1, 0.8f, 0.4f, 1), "Mie 散射（大粒子，气溶胶）");
                ImGui::TextWrapped("相位函数（Henyey-Greenstein）：\n"
                                   "  P_M(θ) = (1-g²) / (4π(1 + g² - 2g·cosθ)^1.5)\n\n"
                                   "  · g = 不对称参数（-1~1）\n"
                                   "    g = 0   → 均匀各向同性散射\n"
                                   "    g = 0.9 → 强烈前向散射（光晕/耀斑效果）\n"
                                   "    g < 0   → 后向散射为主\n\n"
                                   "  · Mie 散射与波长关系：β_M ∝ λ^0（几乎无关）\n"
                                   "    → 所有颜色均匀散射 → 白色光晕");
                ImGui::EndTabItem();
            }

            // ── Tab 2: 参数调节 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("参数调节")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "大气参数实时调节");
                ImGui::Separator();

                bool changed = false;
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 1, 1), "太阳方向：");
                changed |= ImGui::SliderFloat("仰角 (°)", &sunElevation_, -10.0f, 90.0f, "%.1f°");
                changed |= ImGui::SliderFloat("方位角 (°)", &sunAzimuth_, 0.0f, 360.0f, "%.0f°");

                if (sunElevation_ < 0)
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.2f, 1), "  夜晚 / 日落后");
                else if (sunElevation_ < 10)
                    ImGui::TextColored(ImVec4(1, 0.6f, 0.2f, 1), "  日出 / 日落（橙红效果）");
                else if (sunElevation_ < 45)
                    ImGui::TextColored(ImVec4(1, 0.9f, 0.5f, 1), "  午前 / 午后");
                else
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1, 1), "  正午（蓝天）");

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.4f, 0.7f, 1, 1), "Rayleigh 系数（×10⁻⁶）：");
                changed |= ImGui::SliderFloat("β_R 红", &betaR_.x, 0.1f, 10.0f, "%.2f");
                changed |= ImGui::SliderFloat("β_R 绿", &betaR_.y, 0.1f, 10.0f, "%.2f");
                changed |= ImGui::SliderFloat("β_R 蓝", &betaR_.z, 0.1f, 10.0f, "%.2f");

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1, 0.8f, 0.4f, 1), "Mie 系数：");
                changed |= ImGui::SliderFloat("β_M（散射）", &betaM_, 0.001f, 0.05f, "%.4f");
                changed |= ImGui::SliderFloat("g（不对称）", &mieG_, 0.0f, 0.99f, "%.3f");

                if (changed)
                    updateSkyColor();
                ImGui::EndTabItem();
            }

            // ── Tab 3: 颜色预测 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("颜色预测")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "实时天空颜色预测");
                ImGui::Separator();

                ImGui::Text("太阳仰角    : %.1f °", sunElevation_);
                ImGui::Text("太阳方向余弦: %.3f", std::cos((90.0f - sunElevation_) * PI / 180.0f));
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.7f, 0.9f, 1, 1), "天顶颜色（朝上看）：");
                ImGui::ColorButton("天顶##col",
                                   ImVec4(zenithColor_.x, zenithColor_.y, zenithColor_.z, 1),
                                   ImGuiColorEditFlags_NoTooltip,
                                   ImVec2(200, 40));
                ImGui::SameLine();
                ImGui::Text("R=%.2f G=%.2f B=%.2f", zenithColor_.x, zenithColor_.y, zenithColor_.z);

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1, 0.8f, 0.5f, 1), "地平线颜色（水平看）：");
                ImGui::ColorButton("地平线##col",
                                   ImVec4(horizonColor_.x, horizonColor_.y, horizonColor_.z, 1),
                                   ImGuiColorEditFlags_NoTooltip,
                                   ImVec2(200, 40));
                ImGui::SameLine();
                ImGui::Text("R=%.2f G=%.2f B=%.2f", horizonColor_.x, horizonColor_.y, horizonColor_.z);

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1, 0.6f, 0.2f, 1), "太阳方向颜色（看向太阳）：");
                ImGui::ColorButton("太阳##col",
                                   ImVec4(sunColor_.x, sunColor_.y, sunColor_.z, 1),
                                   ImGuiColorEditFlags_NoTooltip,
                                   ImVec2(200, 40));
                ImGui::SameLine();
                ImGui::Text("R=%.2f G=%.2f B=%.2f", sunColor_.x, sunColor_.y, sunColor_.z);

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::TextWrapped("颜色计算原理：\n"
                                   "  · 对每个观察方向，沿视线积分大气散射\n"
                                   "  · 同时积分直射光的 Rayleigh + Mie 贡献\n"
                                   "  · 加入大气衰减（Beer-Lambert 定律）\n"
                                   "  · 最终颜色 = ∫(R散射 + M散射) × 衰减 dt");
                ImGui::EndTabItem();
            }

            // ── Tab 4: 着色器管线 ────────────────────────────────────────
            if (ImGui::BeginTabItem("着色器管线")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "sky_atmo.frag 管线");
                ImGui::Separator();
                ImGui::TextWrapped("管线：单张全屏三角形（3 顶点），无几何体\n\n"
                                   "顶点着色器（sky_atmo.vert）：\n"
                                   "  // 重建视线方向\n"
                                   "  vec3 rayDir = normalize(invViewProj * ndc).xyz;\n\n"
                                   "片元着色器（sky_atmo.frag）：\n"
                                   "  // 输入：\n"
                                   "  //   rayDir  = 视线方向（已归一化）\n"
                                   "  //   sunDir  = 太阳方向（uniformbuffer）\n"
                                   "  //   betaR/M = 散射系数（uniform）\n\n"
                                   "  // 光线步进（16~64步）\n"
                                   "  for (int i = 0; i < STEPS; ++i) {\n"
                                   "      vec3 pos = rayOrigin + rayDir * t;\n"
                                   "      float h = length(pos) - R_EARTH;\n\n"
                                   "      // Rayleigh 和 Mie 密度（指数大气层）\n"
                                   "      float rho_R = exp(-h / H_R);  // H_R = 8000m\n"
                                   "      float rho_M = exp(-h / H_M);  // H_M = 1200m\n\n"
                                   "      // 积分散射贡献\n"
                                   "      rayleigh += rho_R * stepSize;\n"
                                   "      mie      += rho_M * stepSize;\n"
                                   "  }\n\n"
                                   "  // 相位函数 × 系数 → 最终颜色\n"
                                   "  vec3 color = betaR * phaseR * rayleigh\n"
                                   "             + betaM * phaseM * mie;");
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    float sunElevation_ = 30.0f;
    float sunAzimuth_ = 180.0f;
    glm::vec3 betaR_ = {3.8f, 13.5f, 33.1f};
    float betaM_ = 0.021f;
    float mieG_ = 0.76f;

    glm::vec3 zenithColor_ = {0.1f, 0.3f, 0.9f};
    glm::vec3 horizonColor_ = {0.6f, 0.7f, 0.9f};
    glm::vec3 sunColor_ = {1.0f, 0.9f, 0.7f};

    void updateSkyColor() {
        float elevRad = sunElevation_ * PI / 180.0f;
        float cosElev = std::cos(PI / 2.0f - elevRad);
        float t = std::max(0.0f, cosElev);
        float sunset = std::max(0.0f, 1.0f - t * 3.0f);

        zenithColor_.x = 0.05f + t * 0.2f + sunset * 0.3f;
        zenithColor_.y = 0.1f + t * 0.4f - sunset * 0.1f;
        zenithColor_.z = 0.2f + t * 0.85f - sunset * 0.4f;

        horizonColor_.x = 0.4f + t * 0.3f + sunset * 0.5f;
        horizonColor_.y = 0.5f + t * 0.2f + sunset * 0.1f;
        horizonColor_.z = 0.7f + t * 0.2f - sunset * 0.5f;

        sunColor_.x = 1.0f;
        sunColor_.y = 0.95f - sunset * 0.3f;
        sunColor_.z = 0.8f - sunset * 0.6f;

        zenithColor_ = glm::clamp(zenithColor_, glm::vec3(0), glm::vec3(1));
        horizonColor_ = glm::clamp(horizonColor_, glm::vec3(0), glm::vec3(1));
        sunColor_ = glm::clamp(sunColor_, glm::vec3(0), glm::vec3(1));
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第73章：大气散射（Rayleigh + Mie Scattering）\n";
    std::cout << " 后处理特效系列 — ch73/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch73App app;
        app.run("第73章：大气散射");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
