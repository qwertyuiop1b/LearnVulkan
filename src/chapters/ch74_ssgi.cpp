/**
 * @file ch74_ssgi.cpp
 * @brief 第74章：SSGI（屏幕空间全局光照）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【SSAO vs SSGI 对比】
 *
 *  SSAO（屏幕空间环境光遮蔽）：
 *    · 只计算遮蔽（0=被遮蔽，1=开放）
 *    · 输出：灰度图（乘以环境光照）
 *    · 不考虑间接光颜色
 *
 *  SSGI（屏幕空间全局光照）：
 *    · 计算间接光照颜色（彩色）
 *    · 周围表面发出的光对当前点产生贡献
 *    · 输出：RGB 间接辐照度图
 *    · 效果：彩色墙体旁的物体会沾染墙体颜色（Color Bleeding）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <array>
#include <iostream>

class Ch74App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.04f, 0.06f, 0.08f};
    }

    void onUpdate() override {
        estimatePerformance();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第74章：SSGI");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第74章：SSGI（屏幕空间全局光照）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("SsgiTabs")) {

            // ── Tab 1: SSAO vs SSGI ────────────────────────────────────────
            if (ImGui::BeginTabItem("SSAO vs SSGI")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "SSAO 与 SSGI 对比");
                ImGui::Separator();

                ImGui::Columns(2, "compCols");
                ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "SSAO（环境光遮蔽）");
                ImGui::TextWrapped("输出：灰度遮蔽系数\n"
                                   "  0.0 = 完全遮蔽（暗）\n"
                                   "  1.0 = 完全开放（亮）\n\n"
                                   "算法：\n"
                                   "  · 在法线半球内采样 N 个点\n"
                                   "  · 检查点是否被几何体遮挡\n"
                                   "  · 统计遮挡比例\n\n"
                                   "效果：\n"
                                   "  · 凹陷处更暗，接触处更暗\n"
                                   "  · 纯灰度，无颜色信息");
                ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "SSGI（全局光照）");
                ImGui::TextWrapped("输出：彩色间接辐照度\n"
                                   "  RGB = 间接光颜色强度\n\n"
                                   "算法：\n"
                                   "  · G-Buffer：位置 + 法线 + 颜色\n"
                                   "  · 在屏幕空间采样周围片元\n"
                                   "  · 用法线点积加权贡献\n"
                                   "  · 累加周围片元的发射颜色\n\n"
                                   "效果：\n"
                                   "  · 彩色 Color Bleeding\n"
                                   "  · 间接光颜色传递\n"
                                   "  · 近似一次弹射 GI");
                ImGui::Columns(1);
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "SSGI 核心公式：");
                ImGui::TextWrapped("// 对每个像素，在屏幕空间采样周围 N 个点：\n"
                                   "vec3 indirectLight = vec3(0.0);\n"
                                   "for (int i = 0; i < SAMPLE_COUNT; ++i) {\n"
                                   "    vec2 sampleUV = uv + poissonDisk[i] * sampleRadius;\n"
                                   "    vec3 samplePos    = texture(gPosition, sampleUV).xyz;\n"
                                   "    vec3 sampleNormal = texture(gNormal,   sampleUV).xyz;\n"
                                   "    vec3 sampleColor  = texture(gAlbedo,   sampleUV).rgb;\n\n"
                                   "    // 从采样点到当前点的方向\n"
                                   "    vec3 dir = normalize(currentPos - samplePos);\n\n"
                                   "    // Lambertian 加权（法线点积）\n"
                                   "    float NdotL_src  = max(0.0, dot(sampleNormal,  dir));\n"
                                   "    float NdotL_dst  = max(0.0, dot(currentNormal, -dir));\n\n"
                                   "    // 贡献 = 颜色 × 几何权重 / 距离²\n"
                                   "    float dist = length(currentPos - samplePos);\n"
                                   "    indirectLight += sampleColor * NdotL_src * NdotL_dst\n"
                                   "                    / (dist * dist + 0.1);\n"
                                   "}\n"
                                   "indirectLight /= float(SAMPLE_COUNT);");
                ImGui::EndTabItem();
            }

            // ── Tab 2: 质量 vs 性能 ────────────────────────────────────────
            if (ImGui::BeginTabItem("质量 vs 性能")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "采样配置");
                ImGui::Separator();

                const char* sampleOpts[] = {
                    "4 采样（快速，低质量）", "8 采样（游戏推荐）", "16 采样（高质量）", "32 采样（离线质量）"};
                const int sampleVals[] = {4, 8, 16, 32};
                ImGui::Combo("采样数", &sampleIdx_, sampleOpts, 4);

                ImGui::SliderFloat("采样半径（世界空间m）", &sampleRadius_, 0.1f, 5.0f, "%.2f m");
                ImGui::SliderFloat("强度倍率", &giStrength_, 0.0f, 3.0f, "%.2f");
                ImGui::SliderFloat("衰减指数", &falloffPow_, 1.0f, 4.0f, "%.1f");
                ImGui::Checkbox("使用半分辨率", &halfResolution_);
                ImGui::Checkbox("开启时域积累 (TAA)", &useTemporal_);
                ImGui::Spacing();
                ImGui::Separator();

                int samples = sampleVals[sampleIdx_];
                float res = halfResolution_ ? 0.5f : 1.0f;
                ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "性能估算（1920×1080）：");
                ImGui::Text("  G-Buffer Pass     : ~1.5 ms");
                ImGui::Text("  SSGI Pass (%2d采样) : ~%.1f ms", samples, samples * 0.18f * res * res);
                ImGui::Text("  Blur Pass          : ~0.4 ms");
                ImGui::Text("  Temporal 积累      : ~%.1f ms", useTemporal_ ? 0.3f : 0.0f);
                ImGui::Text("  总计               : ~%.1f ms",
                            1.5f + samples * 0.18f * res * res + 0.4f + (useTemporal_ ? 0.3f : 0.0f));
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "质量对比：");
                struct QEntry {
                    int s;
                    const char* q;
                    const char* artifacts;
                };
                std::array<QEntry, 4> entries = {{
                    {4, "低", "明显噪声，需大量模糊"},
                    {8, "中", "轻微噪声，配合 TAA 够用"},
                    {16, "高", "基本无噪声"},
                    {32, "极高", "接近参考质量"},
                }};
                ImGui::Columns(3, "qualCols");
                ImGui::Text("采样数");
                ImGui::NextColumn();
                ImGui::Text("质量");
                ImGui::NextColumn();
                ImGui::Text("噪声情况");
                ImGui::NextColumn();
                ImGui::Separator();
                for (auto& e : entries) {
                    bool cur = (sampleVals[sampleIdx_] == e.s);
                    if (cur)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1, 0.5f, 1));
                    ImGui::Text("%d", e.s);
                    ImGui::NextColumn();
                    ImGui::Text("%s", e.q);
                    ImGui::NextColumn();
                    ImGui::Text("%s", e.artifacts);
                    ImGui::NextColumn();
                    if (cur)
                        ImGui::PopStyleColor();
                }
                ImGui::Columns(1);
                ImGui::EndTabItem();
            }

            // ── Tab 3: Color Bleeding 效果 ────────────────────────────────
            if (ImGui::BeginTabItem("Color Bleeding")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "颜色溢出（Color Bleeding）");
                ImGui::Separator();
                ImGui::TextWrapped("Color Bleeding 是 GI 最直观的效果之一：\n\n"
                                   "  · 红色墙壁旁的白色物体 → 沾染红色\n"
                                   "  · 蓝色地板上的白色盒子底部 → 略带蓝色\n"
                                   "  · 绿色树叶下的地面 → 绿色漫反射\n\n"
                                   "为什么 SSAO 做不到：\n"
                                   "  · SSAO 只输出遮蔽强度（0~1），无颜色信息\n"
                                   "  · 不知道遮蔽来自什么颜色的表面\n\n"
                                   "SSGI 的局限性：\n"
                                   "  · 只能利用屏幕内可见的信息（被相机遮挡的区域无法贡献）\n"
                                   "  · 采样半径有限，只能捕捉近距离的 Color Bleeding\n"
                                   "  · 每帧只计算一次弹射（间接光本身不再弹射）\n\n"
                                   "典型场景效果描述：\n"
                                   "  场景：Cornell Box（白色房间 + 红色左墙 + 绿色右墙）\n"
                                   "  · 左侧球体：沾染红色辉光\n"
                                   "  · 右侧球体：沾染绿色辉光\n"
                                   "  · 中央球体：混合红/绿微弱辉光\n"
                                   "  · 天花板：来自地板的轻微反弹");
                ImGui::Spacing();
                ImGui::SliderFloat("Color Bleeding 强度", &giStrength_, 0.0f, 3.0f, "%.2f");
                ImGui::Text("墙体颜色溢出距离 : ~%.1f m", sampleRadius_);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    int sampleIdx_ = 1;
    float sampleRadius_ = 1.5f;
    float giStrength_ = 1.0f;
    float falloffPow_ = 2.0f;
    bool halfResolution_ = true;
    bool useTemporal_ = true;
    float estimatedMs_ = 0.0f;

    void estimatePerformance() {
        const int sampleVals[] = {4, 8, 16, 32};
        int samples = sampleVals[sampleIdx_];
        float res = halfResolution_ ? 0.5f : 1.0f;
        estimatedMs_ = 1.5f + samples * 0.18f * res * res + 0.4f + (useTemporal_ ? 0.3f : 0.0f);
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第74章：SSGI（屏幕空间全局光照）\n";
    std::cout << " 后处理特效系列 — ch74/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch74App app;
        app.run("第74章：SSGI");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
