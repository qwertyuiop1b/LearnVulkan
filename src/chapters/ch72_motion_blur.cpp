/**
 * @file ch72_motion_blur.cpp
 * @brief 第72章：运动模糊（Per-Object Motion Blur）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【运动模糊原理】
 *
 *  真实相机有快门时间（曝光时间），在曝光期间运动的物体会在底片上留下轨迹。
 *  Per-Object Motion Blur 通过速度缓冲（Velocity Buffer）重建这一效果：
 *
 *  Velocity Buffer（速度缓冲）：
 *    - 每个片元存储从上一帧到当前帧在 NDC 空间的位移
 *    - velocity = (currentNDC.xy - prevNDC.xy) / 2  → UV 空间速度
 *
 *  采样：
 *    - 沿速度方向采样 N 个样点并平均
 *    - 速度越快 → 采样步距越大 → 越模糊
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cmath>
#include <iostream>

class Ch72App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.06f, 0.04f, 0.10f};
        objectAngle_ = 0.0f;
        prevObjectAngle_ = 0.0f;
    }

    void onUpdate() override {
        prevObjectAngle_ = objectAngle_;
        if (isAnimating_) {
            objectAngle_ += angularSpeed_ * 0.016f;
            if (objectAngle_ > 360.0f)
                objectAngle_ -= 360.0f;
        }
        computeVelocity();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第72章：运动模糊");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第72章：运动模糊（Per-Object Motion Blur）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("MotionBlurTabs")) {

            // ── Tab 1: 速度缓冲原理 ──────────────────────────────────────
            if (ImGui::BeginTabItem("速度缓冲原理")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Velocity Buffer — 每对象速度缓冲");
                ImGui::Separator();
                ImGui::TextWrapped("步骤 1 — 顶点着色器中同时输出当前帧和上一帧的 NDC 坐标：\n\n"
                                   "  // 顶点着色器\n"
                                   "  layout(location=0) out vec4 curNDC;   // 当前帧位置\n"
                                   "  layout(location=1) out vec4 prevNDC;  // 上一帧位置\n\n"
                                   "  curNDC  = curMVP  * position;\n"
                                   "  prevNDC = prevMVP * position;\n\n"
                                   "步骤 2 — 片元着色器计算速度（UV 空间）：\n\n"
                                   "  // 将 NDC 转换为 UV 坐标（[0,1] 范围）\n"
                                   "  vec2 curUV  = (curNDC.xy  / curNDC.w)  * 0.5 + 0.5;\n"
                                   "  vec2 prevUV = (prevNDC.xy / prevNDC.w) * 0.5 + 0.5;\n\n"
                                   "  // 速度 = 当前帧 - 上一帧（UV 空间位移）\n"
                                   "  vec2 velocity = curUV - prevUV;\n\n"
                                   "  // 写入 Velocity Buffer（R16G16_SFLOAT）\n"
                                   "  outVelocity = velocity;\n\n"
                                   "步骤 3 — 后处理 Pass 使用速度做模糊：\n\n"
                                   "  vec2 vel = texture(velocityTex, uv).rg;\n"
                                   "  vec4 result = vec4(0);\n"
                                   "  for (int i = 0; i < samples; ++i) {\n"
                                   "      float t = (float(i) / float(samples-1)) - 0.5;\n"
                                   "      result += texture(colorTex, uv + vel * t * strength);\n"
                                   "  }\n"
                                   "  result /= float(samples);");
                ImGui::EndTabItem();
            }

            // ── Tab 2: 参数调节 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("参数调节")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "运动模糊参数");
                ImGui::Separator();

                ImGui::Checkbox("开启动画", &isAnimating_);
                ImGui::SliderFloat("旋转速度 (°/s)", &angularSpeed_, 10.0f, 720.0f, "%.0f°/s");
                ImGui::SliderFloat("模糊强度（×）", &blurStrength_, 0.0f, 4.0f, "%.2f");
                ImGui::Spacing();

                const char* sampleItems[] = {"8 个采样", "12 个采样", "16 个采样", "24 个采样"};
                const int sampleValues[] = {8, 12, 16, 24};
                ImGui::Combo("采样数", &sampleIdx_, sampleItems, 4);
                int samples = sampleValues[sampleIdx_];
                ImGui::Spacing();

                ImGui::Checkbox("快门角度模式", &useShutterAngle_);
                if (useShutterAngle_) {
                    ImGui::SliderFloat("快门角度 (°)", &shutterAngle_, 0.0f, 360.0f, "%.0f°");
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1),
                                       "快门角度 180° = 标准电影感（每帧曝光时间=帧时间/2）");
                }
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "实时统计：");
                ImGui::Text("当前旋转角    : %.1f °", objectAngle_);
                ImGui::Text("帧间角度差    : %.2f °", std::abs(objectAngle_ - prevObjectAngle_));
                ImGui::Text("UV 速度估算   : %.4f", estimatedVelocity_);
                ImGui::Text("模糊像素数    : ~%.1f px", estimatedVelocity_ * 1920.0f * blurStrength_);
                ImGui::Text("当前采样数    : %d", samples);
                ImGui::EndTabItem();
            }

            // ── Tab 3: 速度 vs 模糊量对照表 ─────────────────────────────
            if (ImGui::BeginTabItem("速度 vs 模糊量")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "旋转速度 与 模糊量关系");
                ImGui::Separator();

                struct SpeedEntry {
                    const char* speed;
                    float degPerSec;
                    float velUv;
                    float blurPx;
                };
                std::array<SpeedEntry, 7> table = {{
                    {"慢速  30°/s", 30.0f, 0.0014f, 2.7f},
                    {"正常  60°/s", 60.0f, 0.0028f, 5.3f},
                    {"快速 120°/s", 120.0f, 0.0056f, 10.7f},
                    {"高速 180°/s", 180.0f, 0.0083f, 16.0f},
                    {"极速 360°/s", 360.0f, 0.0167f, 32.1f},
                    {"超速 540°/s", 540.0f, 0.0250f, 48.0f},
                    {"极限 720°/s", 720.0f, 0.0333f, 64.0f},
                }};

                ImGui::Columns(4, "speedTable");
                ImGui::Text("旋转速度");
                ImGui::NextColumn();
                ImGui::Text("UV 速度");
                ImGui::NextColumn();
                ImGui::Text("模糊像素");
                ImGui::NextColumn();
                ImGui::Text("视觉效果");
                ImGui::NextColumn();
                ImGui::Separator();

                for (auto& e : table) {
                    bool highlight = (std::abs(angularSpeed_ - e.degPerSec) < 30.0f);
                    if (highlight)
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1, 0.5f, 1));
                    ImGui::Text("%s", e.speed);
                    ImGui::NextColumn();
                    ImGui::Text("%.4f", e.velUv);
                    ImGui::NextColumn();
                    ImGui::Text("~%.0f", e.blurPx * blurStrength_);
                    ImGui::NextColumn();
                    const char* effect = e.blurPx < 4    ? "清晰"
                                         : e.blurPx < 12 ? "轻微"
                                         : e.blurPx < 24 ? "明显"
                                                         : "强烈";
                    ImGui::Text("%s", effect);
                    ImGui::NextColumn();
                    if (highlight)
                        ImGui::PopStyleColor();
                }
                ImGui::Columns(1);
                ImGui::EndTabItem();
            }

            // ── Tab 4: 采样数 vs 质量 ─────────────────────────────────────
            if (ImGui::BeginTabItem("采样数 vs 质量")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "采样数 vs 质量 / 性能权衡");
                ImGui::Separator();

                struct SampleEntry {
                    int samples;
                    float time;
                    const char* quality;
                };
                std::array<SampleEntry, 6> entries = {{
                    {4, 0.18f, "差（明显条纹）"},
                    {8, 0.32f, "一般（轻微条纹）"},
                    {12, 0.47f, "良好（基本无条纹）"},
                    {16, 0.61f, "优秀（推荐）"},
                    {24, 0.91f, "极佳（高端平台）"},
                    {32, 1.20f, "完美（性能敏感）"},
                }};

                ImGui::Columns(3, "sampleTable");
                ImGui::Text("采样数");
                ImGui::NextColumn();
                ImGui::Text("耗时(ms)");
                ImGui::NextColumn();
                ImGui::Text("质量");
                ImGui::NextColumn();
                ImGui::Separator();
                for (auto& e : entries) {
                    ImGui::Text("%d", e.samples);
                    ImGui::NextColumn();
                    ImGui::Text("%.2f", e.time);
                    ImGui::NextColumn();
                    ImGui::Text("%s", e.quality);
                    ImGui::NextColumn();
                }
                ImGui::Columns(1);
                ImGui::Spacing();
                ImGui::TextWrapped("建议：\n"
                                   "  · 移动端：8 采样 + Temporal AA 补偿\n"
                                   "  · PC 中端：12-16 采样（推荐）\n"
                                   "  · PC 高端：24-32 采样\n"
                                   "  · 可结合 TAA（时域反走样）以少量采样获得好质量");
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    float objectAngle_ = 0.0f;
    float prevObjectAngle_ = 0.0f;
    float angularSpeed_ = 90.0f;
    float blurStrength_ = 1.0f;
    float shutterAngle_ = 180.0f;
    float estimatedVelocity_ = 0.0f;
    int sampleIdx_ = 2;
    bool isAnimating_ = true;
    bool useShutterAngle_ = false;

    void computeVelocity() {
        float deltaAngle = objectAngle_ - prevObjectAngle_;
        float radius = 0.3f;
        float arcLen = std::abs(deltaAngle) * (3.14159f / 180.0f) * radius;
        estimatedVelocity_ = arcLen;
        if (useShutterAngle_) {
            estimatedVelocity_ *= shutterAngle_ / 360.0f;
        }
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第72章：运动模糊（Per-Object Motion Blur）\n";
    std::cout << " 后处理特效系列 — ch72/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch72App app;
        app.run("第72章：运动模糊");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
