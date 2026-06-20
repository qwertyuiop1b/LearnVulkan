/**
 * @file ch71_depth_of_field.cpp
 * @brief 第71章：景深效果（DoF / Depth of Field）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【景深原理】
 *
 *  真实相机的镜头并非针孔，具有有限孔径（光圈），
 *  只有处于焦点距离的物体才会清晰成像。
 *  其他距离的物体在感光元件上形成"弥散圆"（CoC）。
 *
 *  CoC（Circle of Confusion）公式：
 *    CoC = (焦距 × 光圈直径) × |物距 - 焦距| / (物距 × 焦距)
 *  其中 CoC 值越大，模糊程度越高。
 *
 * 【管线描述】
 *  Scene Pass → HDR Color + Depth Buffer
 *  CoC Pass   → Fullscreen，根据深度值计算每像素 CoC 大小
 *  Blur Pass  → Gather 式散景模糊（从 CoC texture 指导模糊半径）
 *  Composite  → 将清晰区域与散景区域按 CoC 权重混合
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <iostream>

class Ch71App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.05f, 0.05f, 0.12f};
    }

    void onUpdate() override {
        recomputeCoC();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第71章：景深效果（DoF）");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第71章：景深效果（Depth of Field）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("DoFTabs")) {

            // ── Tab 1: CoC 公式与参数 ──────────────────────────────────────
            if (ImGui::BeginTabItem("CoC 公式与参数")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "弥散圆（Circle of Confusion）");
                ImGui::Separator();
                ImGui::TextWrapped("CoC = (焦距 × 光圈直径) × |物距 - 焦点距离| / (物距 × 焦点距离)\n\n"
                                   "  · CoC > 0    → 物体在焦点后方（背景虚化）\n"
                                   "  · CoC < 0    → 物体在焦点前方（前景虚化）\n"
                                   "  · |CoC| > 阈值 → 开始模糊处理\n\n"
                                   "  归一化后（像素单位）：\n"
                                   "    cocPixels = CoC × (sensorHeight / imageHeight)");
                ImGui::Spacing();
                ImGui::Separator();

                ImGui::SliderFloat("焦点距离 (m)", &focusDistance_, 0.5f, 50.0f, "%.2f m");
                ImGui::SliderFloat("焦距 (mm)", &focalLength_, 10.0f, 200.0f, "%.1f mm");
                ImGui::SliderFloat("光圈 (f/)", &aperture_, 1.0f, 22.0f, "f/%.1f");
                ImGui::SliderFloat("最大散景半径", &maxBokehRadius_, 2.0f, 32.0f, "%.0f px");
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.5f, 1), "实时计算结果：");
                ImGui::Text("  焦点处  CoC = 0.00 px（始终清晰）");
                ImGui::Text("  前景 @  1.0m  CoC = %.2f px", computeCoC(1.0f));
                ImGui::Text("  背景 @  10.0m CoC = %.2f px", computeCoC(10.0f));
                ImGui::Text("  背景 @  ∞     CoC = %.2f px", computeCoC(1000.0f));
                ImGui::Spacing();

                float nearCoc = std::abs(computeCoC(focusDistance_ * 0.5f));
                float farCoc = std::abs(computeCoC(focusDistance_ * 2.0f));
                ImGui::Text("  前景虚化量（%.1fm）: %.2f px", focusDistance_ * 0.5f, nearCoc);
                ImGui::ProgressBar(nearCoc / maxBokehRadius_, ImVec2(-1, 16));
                ImGui::Text("  背景虚化量（%.1fm）: %.2f px", focusDistance_ * 2.0f, farCoc);
                ImGui::ProgressBar(farCoc / maxBokehRadius_, ImVec2(-1, 16));
                ImGui::EndTabItem();
            }

            // ── Tab 2: 散景形状 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("散景形状")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Bokeh 形状选项");
                ImGui::Separator();

                const char* shapes[] = {"圆形（Poisson Disk）", "六边形（光圈叶片）"};
                ImGui::Combo("散景形状", &bokehShape_, shapes, 2);
                ImGui::Spacing();

                if (bokehShape_ == 0) {
                    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "圆形散景 — Poisson 圆盘采样");
                    ImGui::TextWrapped("原理：在圆形区域内，用 Poisson 分布预生成采样点，\n"
                                       "保证采样均匀（避免聚集），无明显规律性噪声。\n\n"
                                       "实现：\n"
                                       "  // 预生成 N 个 Poisson 圆盘采样点\n"
                                       "  const vec2 poissonDisk[16] = { ... };\n\n"
                                       "  vec4 result = vec4(0);\n"
                                       "  float coc = texture(cocTex, uv).r;  // 当前像素 CoC 大小\n"
                                       "  for (int i = 0; i < 16; ++i) {\n"
                                       "      vec2 offset = poissonDisk[i] * coc * bokehScale;\n"
                                       "      result += texture(hdrTex, uv + offset);\n"
                                       "  }\n"
                                       "  result /= 16.0;\n\n"
                                       "优点：各向同性，符合镜头自然散景\n"
                                       "缺点：采样数有限时有噪声");
                } else {
                    ImGui::TextColored(ImVec4(1, 0.6f, 0.3f, 1), "六边形散景 — 光圈叶片模拟");
                    ImGui::TextWrapped("原理：真实相机光圈由多片叶片组成（通常 5~9 片），\n"
                                       "叶片数决定散景形状（6 片叶片 → 六边形光斑）。\n\n"
                                       "实现：分三次定向模糊\n"
                                       "  Pass A: 沿 0°  方向做线性模糊\n"
                                       "  Pass B: 沿 60° 方向做线性模糊（基于 Pass A）\n"
                                       "  Pass C: 沿 120° 方向做线性模糊（基于 Pass A）\n"
                                       "  最终结果 = 混合 Pass B + Pass C\n\n"
                                       "  vec2 dir60  = vec2(0.866, 0.5);    // cos60, sin60\n"
                                       "  vec2 dir120 = vec2(-0.866, 0.5);   // cos120, sin120\n\n"
                                       "优点：性能较好，形状有特色感\n"
                                       "缺点：方向性明显，不如圆形自然");
                }
                ImGui::Spacing();
                ImGui::SliderInt("采样数", &sampleCount_, 8, 64);
                ImGui::SliderFloat("散景强度", &bokehStrength_, 0.0f, 2.0f, "%.2f");
                ImGui::EndTabItem();
            }

            // ── Tab 3: 渲染管线 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("渲染管线")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "DoF 完整渲染管线");
                ImGui::Separator();
                ImGui::TextWrapped("Pass 1 — Scene Pass\n"
                                   "  输出：HDR Color Texture + Depth Buffer\n"
                                   "  格式：R16G16B16A16_SFLOAT（HDR） + D32_SFLOAT（深度）\n\n"
                                   "Pass 2 — CoC Pass（全屏后处理）\n"
                                   "  输入：Depth Buffer\n"
                                   "  计算：对每个像素根据深度值计算 CoC 大小\n"
                                   "  输出：CoC Texture（R16_SFLOAT，正值=背景，负值=前景）\n"
                                   "  着色器：coc.frag\n\n"
                                   "Pass 3 — Downscale + Blur Pass\n"
                                   "  将 HDR 纹理缩小到 1/2 分辨率（减少带宽）\n"
                                   "  按 CoC 大小做散景采样（gather 式）\n"
                                   "  输出：模糊后的散景 Texture（1/2 分辨率）\n\n"
                                   "Pass 4 — Composite Pass\n"
                                   "  输入：原始清晰纹理 + 模糊散景纹理 + CoC\n"
                                   "  混合权重：clamp(|coc| / cocThreshold, 0, 1)\n"
                                   "  输出：最终图像\n\n"
                                   "  // 合成着色器核心：\n"
                                   "  float blendFactor = saturate(abs(coc) / cocThreshold);\n"
                                   "  vec3 final = mix(sharpColor, bokehColor, blendFactor);");
                ImGui::EndTabItem();
            }

            // ── Tab 4: 统计展示 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("统计展示")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "焦深（Depth of Field）统计");
                ImGui::Separator();

                float cocNear = computeCoCThreshold();
                ImGui::Text("焦点距离        : %.2f m", focusDistance_);
                ImGui::Text("前焦深边界      : %.2f m", focusDistance_ - cocNear);
                ImGui::Text("后焦深边界      : %.2f m", focusDistance_ + cocNear * 1.5f);
                ImGui::Text("焦深范围        : %.2f m", cocNear * 2.5f);
                ImGui::Spacing();
                ImGui::Text("光圈大小        : f/%.1f", aperture_);
                ImGui::Text("散景最大半径    : %.0f px", maxBokehRadius_);
                ImGui::Text("CoC @焦点前2m   : %.2f px", std::abs(computeCoC(focusDistance_ - 2.0f)));
                ImGui::Text("CoC @焦点后5m   : %.2f px", std::abs(computeCoC(focusDistance_ + 5.0f)));
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "性能估算（1920×1080）：");
                ImGui::Text("  CoC Pass       : ~0.2 ms");
                ImGui::Text("  Downscale      : ~0.3 ms");
                ImGui::Text("  Blur Pass(%2d) : ~%.1f ms", sampleCount_, sampleCount_ * 0.12f);
                ImGui::Text("  Composite      : ~0.15 ms");
                ImGui::Text("  总计           : ~%.1f ms", 0.65f + sampleCount_ * 0.12f);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    float focusDistance_ = 5.0f;
    float focalLength_ = 50.0f;
    float aperture_ = 2.8f;
    float maxBokehRadius_ = 16.0f;
    float bokehStrength_ = 1.0f;
    int bokehShape_ = 0;
    int sampleCount_ = 16;

    /// @brief 计算指定物距处的 CoC 像素大小
    float computeCoC(float objectDistance) const {
        if (objectDistance <= 0.0f)
            return 0.0f;
        const float f = focalLength_ * 0.001f;
        const float D = f / aperture_;
        const float fd = focusDistance_;
        float coc = D * std::abs(objectDistance - fd) / (objectDistance * fd) * f;
        return std::min(coc * 1000.0f, maxBokehRadius_);
    }

    /// @brief 估算焦深边界（CoC 小于 1 像素的范围）
    float computeCoCThreshold() const {
        const float f = focalLength_ * 0.001f;
        const float D = f / aperture_;
        return focusDistance_ * focusDistance_ / (D * f * 1000.0f - focusDistance_);
    }

    void recomputeCoC() {}
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第71章：景深效果（Depth of Field）\n";
    std::cout << " 后处理特效系列 — ch71/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch71App app;
        app.run("第71章：景深效果（DoF）");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
