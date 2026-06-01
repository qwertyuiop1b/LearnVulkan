/**
 * @file ch87_parallax_occlusion.cpp
 * @brief 第87章：视差遮蔽贴图（Parallax Occlusion Mapping / POM）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【层次对比】
 *
 *  Normal Mapping  — 仅改变法线，不改变轮廓
 *  Parallax Mapping — 简单 UV 偏移，无自遮挡
 *  POM             — 射线步进 + 二分精修，真实自遮挡
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <array>
#include <iostream>

class Ch87App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.04f, 0.05f, 0.07f};
    }

    void onUpdate() override {
        if (animateView_)
            viewAngle_ += 0.4f * 0.016f;
        simulatePom();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第87章：视差遮蔽贴图");
        ImGui::Separator();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第87章：视差遮蔽贴图（POM）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("PomTabs")) {
            if (ImGui::BeginTabItem("算法对比")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "三种技术对比");
                ImGui::Separator();
                const char* modes[] = {"法线贴图", "视差映射", "POM（射线步进）"};
                ImGui::Combo("当前模式", &mappingMode_, modes, 3);
                ImGui::TextWrapped("POM 射线步进（pom.frag）：\n\n"
                                   "  vec3 V = normalize(cameraPos - worldPos);\n"
                                   "  vec3 Vt = normalize(TBN * V);\n"
                                   "  float stepSize = 1.0 / numSteps;\n"
                                   "  for (int i = 0; i < numSteps; ++i) {\n"
                                   "      float layerDepth = i * stepSize;\n"
                                   "      vec2 uv = texCoord - Vt.xy * layerDepth * heightScale;\n"
                                   "      float sampledH = texture(heightMap, uv).r;\n"
                                   "      if (layerDepth >= sampledH) break;  // 命中\n"
                                   "  }\n"
                                   "  // 二分精修提高精度");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("参数调节")) {
                ImGui::SliderFloat("高度缩放", &heightScale_, 0.01f, 0.2f, "%.3f");
                ImGui::SliderInt("步进次数", &numSteps_, 4, 64);
                ImGui::SliderInt("二分迭代", &refineSteps_, 0, 8);
                ImGui::SliderFloat("观察角度", &viewAngle_, 0.0f, 6.28f, "%.2f rad");
                ImGui::Checkbox("自动旋转视角", &animateView_);
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "模拟结果：");
                ImGui::Text("  UV 偏移量   : %.4f", uvOffset_);
                ImGui::Text("  命中层深度  : %.4f", hitDepth_);
                ImGui::Text("  自遮挡检测  : %s", selfOccluded_ ? "是" : "否");
                ImGui::Text("  估算 ALU 开销: %d ops/px", aluCost_);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("高度剖面预览")) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                ImVec2 size(420, 200);
                dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(20, 25, 35, 255));
                float brickH = 0.08f;
                float mortarH = 0.0f;
                for (int i = 0; i < 40; ++i) {
                    float t = float(i) / 40.0f;
                    float h = (int(i / 5) % 2 == 0) ? brickH : mortarH;
                    h += 0.02f * std::sin(t * 20.0f);
                    float px = pos.x + t * size.x;
                    float py = pos.y + size.y - h * size.y * 3.0f;
                    dl->AddLine(ImVec2(px, pos.y + size.y), ImVec2(px, py), IM_COL32(180, 120, 80, 255), 2.0f);
                }
                float rayX = pos.x + (0.5f + 0.3f * std::cos(viewAngle_)) * size.x;
                float rayY = pos.y + 10.0f;
                dl->AddLine(ImVec2(rayX, rayY),
                            ImVec2(rayX - uvOffset_ * size.x * 5, pos.y + size.y - hitDepth_ * size.y),
                            IM_COL32(255, 255, 100, 200),
                            2.0f);
                ImGui::Dummy(size);
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    int mappingMode_ = 2;
    float heightScale_ = 0.05f;
    int numSteps_ = 24;
    int refineSteps_ = 4;
    float viewAngle_ = 0.8f;
    bool animateView_ = true;
    float uvOffset_ = 0.0f;
    float hitDepth_ = 0.0f;
    bool selfOccluded_ = false;
    int aluCost_ = 0;

    void simulatePom() {
        float viewTilt = std::abs(std::sin(viewAngle_));
        switch (mappingMode_) {
        case 0:
            uvOffset_ = 0.0f;
            hitDepth_ = 0.0f;
            selfOccluded_ = false;
            aluCost_ = 8;
            break;
        case 1:
            uvOffset_ = heightScale_ * viewTilt * 0.5f;
            hitDepth_ = heightScale_;
            selfOccluded_ = false;
            aluCost_ = 12;
            break;
        default:
            uvOffset_ = heightScale_ * viewTilt;
            hitDepth_ = heightScale_ * 0.6f;
            selfOccluded_ = viewTilt > 0.3f;
            aluCost_ = numSteps_ * 4 + refineSteps_ * 6 + 10;
            break;
        }
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第87章：视差遮蔽贴图（POM）\n";
    std::cout << " 材质渲染系列 — ch87/6\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch87App app;
        app.run("第87章：视差遮蔽贴图");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
