/**
 * @file ch75_lut_color_grading.cpp
 * @brief 第75章：LUT 色彩分级（3D Look-Up Table）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【什么是 3D LUT】
 *
 *  LUT（Look-Up Table）= 色彩变换表：
 *    输入：原始 RGB 颜色
 *    输出：分级后的 RGB 颜色
 *
 *  3D LUT（三维查找表）：
 *    · 尺寸通常为 16×16×16 或 32×32×32
 *    · 存储为 3D 纹理（VkImageType = VK_IMAGE_TYPE_3D）
 *    · 查找：texture(lut3D, inputRGB) → outputRGB
 *    · 可以表示任意非线性色彩变换
 *
 *  恒等 LUT（Identity LUT）：
 *    · outputRGB = inputRGB（无变换）
 *    · 这是调色的起点，可以在 DCC 工具中修改
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <array>
#include <string>
#include <iostream>

class Ch75App : public DemoApp {
protected:
    void onInit() override
    {
        bgColor_ = {0.06f, 0.05f, 0.08f};
    }

    void onUpdate() override {}

    void buildUi() override
    {
        interactive_.buildDebugPanel("第75章：LUT 色彩分级");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第75章：LUT 色彩分级（3D Look-Up Table）", nullptr))
        { ImGui::End(); return; }

        if (ImGui::BeginTabBar("LutTabs")) {

            // ── Tab 1: 什么是 3D LUT ──────────────────────────────────────
            if (ImGui::BeginTabItem("3D LUT 原理")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "3D Look-Up Table 工作原理");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "3D LUT 是一个三维查找表，将输入 RGB 映射到输出 RGB：\n\n"
                    "  · 尺寸：16×16×16（4096 个条目）\n"
                    "  · 存储：VkImage（VK_IMAGE_TYPE_3D）\n"
                    "  · 格式：VK_FORMAT_R8G8B8A8_SRGB\n\n"
                    "  // 片元着色器（lut_grade.frag）：\n"
                    "  vec3 inputColor = texture(hdrTex, uv).rgb;\n\n"
                    "  // 线性化（如果需要从 sRGB 到线性）\n"
                    "  inputColor = pow(inputColor, vec3(2.2));\n\n"
                    "  // 将输入范围[0,1]映射到LUT坐标（避免边界溢出）\n"
                    "  const float LUT_SIZE = 16.0;\n"
                    "  vec3 lutCoord = inputColor * (LUT_SIZE-1.0)/LUT_SIZE\n"
                    "                 + 0.5/LUT_SIZE;\n\n"
                    "  // 查找 LUT（硬件三线性插值）\n"
                    "  vec3 gradedColor = texture(lut3D, lutCoord).rgb;\n\n"
                    "  // blendAmount 控制 LUT 效果强度\n"
                    "  outColor = mix(inputColor, gradedColor, blendAmount);");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "VkImage 创建参数：");
                ImGui::TextWrapped(
                    "  imageCI.imageType   = VK_IMAGE_TYPE_3D;\n"
                    "  imageCI.format      = VK_FORMAT_R8G8B8A8_SRGB;\n"
                    "  imageCI.extent      = {16, 16, 16};\n"
                    "  imageCI.mipLevels   = 1;\n"
                    "  imageCI.arrayLayers = 1;\n"
                    "  imageCI.usage       = VK_IMAGE_USAGE_SAMPLED_BIT\n"
                    "                       | VK_IMAGE_USAGE_TRANSFER_DST_BIT;\n\n"
                    "  // 采样器：开启三线性插值\n"
                    "  samplerCI.minFilter  = VK_FILTER_LINEAR;\n"
                    "  samplerCI.magFilter  = VK_FILTER_LINEAR;\n"
                    "  samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;");
                ImGui::EndTabItem();
            }

            // ── Tab 2: 预设风格 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("预设风格")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "LUT 预设切换");
                ImGui::Separator();

                const char* presetNames[] = {
                    "原始（无 LUT）",
                    "电影（高对比，低饱和）",
                    "夜视（绿色单色）",
                    "暖调（橙红偏移）",
                    "冷调（蓝绿偏移）",
                };
                ImGui::Text("当前预设：");
                for (int i = 0; i < 5; ++i) {
                    if (ImGui::RadioButton(presetNames[i], &selectedPreset_, i))
                        applyPreset(i);
                    if (i < 4) ImGui::SameLine();
                }
                ImGui::Spacing();
                ImGui::SliderFloat("混合强度 (blendAmount)", &blendAmount_, 0.0f, 1.0f, "%.2f");
                ImGui::Spacing();
                ImGui::Separator();

                // 显示预设颜色样本
                ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "颜色样本预览（输入 → 输出）：");
                ImGui::Columns(4, "colorSamples");
                ImGui::Text("输入颜色"); ImGui::NextColumn();
                ImGui::Text("原始输出"); ImGui::NextColumn();
                ImGui::Text("LUT 输出"); ImGui::NextColumn();
                ImGui::Text("混合结果"); ImGui::NextColumn();
                ImGui::Separator();

                struct SampleColor { glm::vec3 input; const char* name; };
                std::array<SampleColor, 5> samples = {{
                    {{1.0f, 0.2f, 0.2f}, "红色"},
                    {{0.2f, 1.0f, 0.2f}, "绿色"},
                    {{0.2f, 0.2f, 1.0f}, "蓝色"},
                    {{1.0f, 1.0f, 0.2f}, "黄色"},
                    {{0.9f, 0.9f, 0.9f}, "白色"},
                }};

                for (auto& s : samples) {
                    glm::vec3 lutOut = applyLutTransform(s.input);
                    glm::vec3 mixed  = glm::mix(s.input, lutOut, blendAmount_);
                    ImGui::ColorButton((std::string("##in_") + s.name).c_str(),
                        ImVec4(s.input.x, s.input.y, s.input.z, 1),
                        ImGuiColorEditFlags_NoTooltip, ImVec2(60, 20));
                    ImGui::NextColumn();
                    ImGui::ColorButton((std::string("##orig_") + s.name).c_str(),
                        ImVec4(s.input.x, s.input.y, s.input.z, 1),
                        ImGuiColorEditFlags_NoTooltip, ImVec2(60, 20));
                    ImGui::NextColumn();
                    ImGui::ColorButton((std::string("##lut_") + s.name).c_str(),
                        ImVec4(lutOut.x, lutOut.y, lutOut.z, 1),
                        ImGuiColorEditFlags_NoTooltip, ImVec2(60, 20));
                    ImGui::NextColumn();
                    ImGui::ColorButton((std::string("##mix_") + s.name).c_str(),
                        ImVec4(mixed.x, mixed.y, mixed.z, 1),
                        ImGuiColorEditFlags_NoTooltip, ImVec2(60, 20));
                    ImGui::NextColumn();
                }
                ImGui::Columns(1);
                ImGui::EndTabItem();
            }

            // ── Tab 3: 制作方式 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("LUT 制作方式")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "如何制作 3D LUT");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "步骤 1 — 生成恒等 LUT\n"
                    "  恒等 LUT：输出 = 输入（即「不做任何变换」）\n\n"
                    "  // 生成 16×16×16 恒等 LUT 数据：\n"
                    "  for (int b = 0; b < 16; ++b)\n"
                    "  for (int g = 0; g < 16; ++g)\n"
                    "  for (int r = 0; r < 16; ++r) {\n"
                    "      data[b][g][r] = {\n"
                    "          uint8_t(r * 255 / 15),   // R\n"
                    "          uint8_t(g * 255 / 15),   // G\n"
                    "          uint8_t(b * 255 / 15),   // B\n"
                    "          255                       // A\n"
                    "      };\n"
                    "  }\n\n"
                    "步骤 2 — 在 DCC 工具中调色\n"
                    "  · 将恒等 LUT 截图导入 DaVinci Resolve / Photoshop\n"
                    "  · 添加颜色调整层（对比度/色温/色相偏移等）\n"
                    "  · 导出为 .cube 文件（文本格式的 LUT）\n\n"
                    "步骤 3 — 解析 .cube 文件\n"
                    "  .cube 格式：\n"
                    "    LUT_3D_SIZE 16\n"
                    "    # 16^3 = 4096 行 RGB 数据\n"
                    "    0.000 0.000 0.000\n"
                    "    0.020 0.000 0.000\n"
                    "    ...\n\n"
                    "步骤 4 — 上传到 GPU\n"
                    "  · 创建 VkImage（3D，16×16×16）\n"
                    "  · 通过 Staging Buffer 上传数据\n"
                    "  · 布局转换：UNDEFINED → SHADER_READ_ONLY_OPTIMAL");
                ImGui::EndTabItem();
            }

            // ── Tab 4: 参数与性能 ─────────────────────────────────────────
            if (ImGui::BeginTabItem("参数与性能")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "色彩分级参数");
                ImGui::Separator();

                ImGui::SliderFloat("曝光补偿 (EV)", &exposure_,    -3.0f, 3.0f,  "%.2f EV");
                ImGui::SliderFloat("对比度",         &contrast_,    0.5f,  2.0f,  "%.2f");
                ImGui::SliderFloat("饱和度",         &saturation_,  0.0f,  2.0f,  "%.2f");
                ImGui::SliderFloat("色温 (K)",       &temperature_, 2000.0f, 10000.0f, "%.0f K");
                ImGui::SliderFloat("LUT 混合",       &blendAmount_, 0.0f,  1.0f,  "%.2f");
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "性能（极低开销）：");
                ImGui::Text("  LUT 纹理大小    : 16×16×16 × 4B = 16 KB");
                ImGui::Text("  Pass 类型       : Fullscreen 后处理");
                ImGui::Text("  GPU 耗时        : ~0.1–0.2 ms");
                ImGui::Text("  带宽需求        : 极低（3D 纹理缓存友好）");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "LUT 尺寸对比：");
                ImGui::Columns(3, "lutSizeCols");
                ImGui::Text("尺寸");  ImGui::NextColumn();
                ImGui::Text("精度"); ImGui::NextColumn();
                ImGui::Text("大小");  ImGui::NextColumn();
                ImGui::Separator();
                ImGui::Text("16³");  ImGui::NextColumn(); ImGui::Text("标准"); ImGui::NextColumn(); ImGui::Text("16 KB");  ImGui::NextColumn();
                ImGui::Text("32³");  ImGui::NextColumn(); ImGui::Text("高");   ImGui::NextColumn(); ImGui::Text("128 KB"); ImGui::NextColumn();
                ImGui::Text("64³");  ImGui::NextColumn(); ImGui::Text("极高"); ImGui::NextColumn(); ImGui::Text("1 MB");   ImGui::NextColumn();
                ImGui::Columns(1);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

private:
    int   selectedPreset_ = 0;
    float blendAmount_    = 1.0f;
    float exposure_       = 0.0f;
    float contrast_       = 1.0f;
    float saturation_     = 1.0f;
    float temperature_    = 6500.0f;

    // LUT 变换参数（每个预设对应不同的变换）
    glm::vec3 lutLift_   = {0, 0, 0};
    glm::vec3 lutGamma_  = {1, 1, 1};
    glm::vec3 lutGain_   = {1, 1, 1};

    void applyPreset(int preset)
    {
        switch (preset) {
        case 0: // 原始
            lutLift_ = {0,0,0}; lutGamma_ = {1,1,1}; lutGain_ = {1,1,1};
            break;
        case 1: // 电影
            lutLift_ = {0.02f, 0.01f, 0.0f};
            lutGamma_ = {0.9f, 0.85f, 0.8f};
            lutGain_  = {1.05f, 1.0f, 0.95f};
            break;
        case 2: // 夜视
            lutLift_ = {0,0,0};
            lutGamma_ = {0.3f, 1.0f, 0.3f};
            lutGain_  = {0.3f, 1.1f, 0.3f};
            break;
        case 3: // 暖调
            lutLift_ = {0.03f, 0.01f, -0.02f};
            lutGamma_ = {1.1f, 1.0f, 0.85f};
            lutGain_  = {1.1f, 1.0f, 0.8f};
            break;
        case 4: // 冷调
            lutLift_ = {-0.02f, 0.01f, 0.04f};
            lutGamma_ = {0.85f, 0.95f, 1.15f};
            lutGain_  = {0.85f, 1.0f, 1.2f};
            break;
        default: break;
        }
    }

    /// @brief 模拟 LUT 变换（Lift/Gamma/Gain ASC CDL）
    glm::vec3 applyLutTransform(glm::vec3 color) const
    {
        color = color * lutGain_ + lutLift_;
        color.x = std::pow(std::max(0.0f, color.x), 1.0f / lutGamma_.x);
        color.y = std::pow(std::max(0.0f, color.y), 1.0f / lutGamma_.y);
        color.z = std::pow(std::max(0.0f, color.z), 1.0f / lutGamma_.z);
        return glm::clamp(color, glm::vec3(0), glm::vec3(1));
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第75章：LUT 色彩分级（3D Look-Up Table）\n";
    std::cout << " 后处理特效系列 — ch75/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch75App app;
        app.run("第75章：LUT 色彩分级");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
