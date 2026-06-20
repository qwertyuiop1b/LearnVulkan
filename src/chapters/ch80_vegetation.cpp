/**
 * @file ch80_vegetation.cpp
 * @brief 第80章：植被渲染（草地 + 树木）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【草地系统】
 *
 *  每根草叶由 6 个顶点组成（3 个三角形，弯曲叶片）：
 *    顶点 0-1：草根（固定）
 *    顶点 2-3：叶中
 *    顶点 4-5：叶尖（最大摆动量）
 *
 *  GPU 生成（Compute Shader）：
 *    · 根据密度在 SSBO 中生成 blade 数据
 *    · 位置：(x, heightmap(x,z), z)
 *    · 随机参数：高度 / 宽度 / 朝向 / 倾斜
 *
 *  风力动画（vertex shader）：
 *    float windPhase = dot(blade.position.xz, windDir) * windFreq + time;
 *    float windStrength = sin(windPhase) * height * windAmplitude;
 *    // 越高处摆动越大（乘以高度权重）
 *    offset.xz += windDir * windStrength * (vertexT * vertexT);
 *
 * 【草叶视锥剔除（Compute Shader）】
 *
 *  · 读取所有 blade 的中心位置
 *  · 对 6 个裁剪平面进行球形测试
 *  · 通过的 blade 写入 indirect draw buffer
 *  · 使用 vkCmdDrawIndirect 渲染（GPU 自决定数量）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <iostream>

class Ch80App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.12f, 0.18f, 0.08f};
        rebuildSimulation();
    }

    void onUpdate() override {
        windTime_ += 0.016f;
        if (isSimulating_) {
            simulateCulling();
        }
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第80章：植被渲染");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第80章：植被渲染（草地 + 树木）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("VegetationTabs")) {

            // ── Tab 1: 草地系统 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("草地系统")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "GPU 草地渲染系统");
                ImGui::Separator();
                ImGui::TextWrapped("草叶几何（每叶 6 顶点 / 3 三角形）：\n\n"
                                   "  // Blade 数据结构（存在 SSBO）：\n"
                                   "  struct GrassBlade {\n"
                                   "      vec4 positionAndHeight;  // xyz=位置, w=高度\n"
                                   "      vec4 directionAndWidth;  // xy=朝向, z=宽度, w=随机\n"
                                   "      vec4 tilt;               // 倾斜向量\n"
                                   "  };\n\n"
                                   "  // 顶点着色器：将 SSBO 数据 + vertexID 转换为位置\n"
                                   "  int bladeIdx = gl_InstanceIndex;\n"
                                   "  int vertIdx  = gl_VertexIndex;  // 0-5\n"
                                   "  GrassBlade blade = blades[bladeIdx];\n\n"
                                   "  // 顶点位置（弯曲叶片，贝塞尔曲线）：\n"
                                   "  float t = float(vertIdx / 2) / 2.0;  // 0, 0.5, 1\n"
                                   "  vec3 pos = blade.position.xyz\n"
                                   "           + blade.tilt.xyz * t          // 倾斜\n"
                                   "           + vec3(0, blade.w * t, 0);    // 高度\n\n"
                                   "风力动画：\n"
                                   "  vec2 windDir = normalize(vec2(cos(windAngle), sin(windAngle)));\n"
                                   "  float phase  = dot(pos.xz, windDir) * 0.1 + time * 2.0;\n"
                                   "  float swing  = sin(phase) * windStrength * t * t;\n"
                                   "  pos.xz += windDir * swing;  // 只有叶尖大幅摆动");
                ImGui::EndTabItem();
            }

            // ── Tab 2: 草叶剔除 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("Compute 剔除")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Compute Shader 视锥剔除");
                ImGui::Separator();
                ImGui::TextWrapped("// 草叶剔除 Compute Shader：grass_cull.comp\n"
                                   "// 每个工作组处理 64 根草叶\n"
                                   "layout(local_size_x=64, local_size_y=1, local_size_z=1) in;\n\n"
                                   "void main() {\n"
                                   "    uint idx = gl_GlobalInvocationID.x;\n"
                                   "    if (idx >= totalBlades) return;\n\n"
                                   "    GrassBlade blade = blades[idx];\n"
                                   "    vec3 center = blade.positionAndHeight.xyz\n"
                                   "               + vec3(0, blade.positionAndHeight.w * 0.5, 0);\n"
                                   "    float radius = blade.positionAndHeight.w * 0.5;\n\n"
                                   "    // 对 6 个视锥平面做球形测试\n"
                                   "    bool visible = true;\n"
                                   "    for (int i = 0; i < 6; ++i) {\n"
                                   "        float dist = dot(frustumPlanes[i].xyz, center)\n"
                                   "                   + frustumPlanes[i].w;\n"
                                   "        if (dist < -radius) { visible = false; break; }\n"
                                   "    }\n\n"
                                   "    if (visible) {\n"
                                   "        // Atomic 计数 + 写入 indirect buffer\n"
                                   "        uint slot = atomicAdd(indirectCmd.instanceCount, 1);\n"
                                   "        visibleIndices[slot] = idx;\n"
                                   "    }\n"
                                   "}");
                ImGui::EndTabItem();
            }

            // ── Tab 3: 树木渲染 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("树木渲染")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "树木 LOD 渲染");
                ImGui::Separator();
                ImGui::TextWrapped("简化树木模型（程序化生成）：\n\n"
                                   "  树干：圆柱体\n"
                                   "    半径：0.1m ~ 0.5m\n"
                                   "    高度：2m ~ 10m\n"
                                   "    顶点数：16 边 × 2 环 = 32 顶点\n\n"
                                   "  树冠：球形（近似）\n"
                                   "    中心：树干顶部\n"
                                   "    半径：树干高度 × 0.6\n"
                                   "    颜色：深绿 ~ 浅绿（随机偏移）\n\n"
                                   "LOD 切换策略：\n"
                                   "  距离 < 30m  → 完整 3D 模型（树干+球形树冠）\n"
                                   "  30m ~ 80m  → 简化模型（少边多边形）\n"
                                   "  80m ~ 200m → 2D Billboard（始终朝向相机）\n"
                                   "  > 200m     → 不渲染（太远不可见）\n\n"
                                   "Billboard 实现：\n"
                                   "  // vertex shader\n"
                                   "  vec3 camRight = normalize(cross(camDir, vec3(0,1,0)));\n"
                                   "  vec3 camUp    = vec3(0, 1, 0);\n"
                                   "  // 四边形顶点 = 树中心 ± right × width/2 ± up × height/2\n"
                                   "  vec3 pos = treeCenter\n"
                                   "           + camRight * (uv.x - 0.5) * treeWidth\n"
                                   "           + camUp    * uv.y * treeHeight;");
                ImGui::EndTabItem();
            }

            // ── Tab 4: 性能参数 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("性能参数")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "植被性能配置");
                ImGui::Separator();

                bool changed = false;
                const char* densityOpts[] = {"100 根草叶", "1000 根草叶", "10000 根草叶", "100000 根草叶"};
                const int densityVals[] = {100, 1000, 10000, 100000};
                changed |= ImGui::Combo("草叶密度", &densityIdx_, densityOpts, 4);
                changed |= ImGui::SliderFloat("渲染面积 (m)", &renderArea_, 10.0f, 200.0f, "%.0f m");
                ImGui::Checkbox("开启 Compute 剔除", &enableCulling_);
                ImGui::Checkbox("动态风力", &isSimulating_);
                ImGui::SliderFloat("风力强度", &windStrength_, 0.0f, 2.0f, "%.2f");
                ImGui::SliderFloat("风向角 (°)", &windAngle_, 0.0f, 360.0f, "%.0f°");
                if (changed)
                    rebuildSimulation();

                ImGui::Spacing();
                ImGui::Separator();

                int totalBlades = densityVals[densityIdx_];
                int visibleBlades = enableCulling_ ? culledVisibleCount_ : totalBlades;
                int cullRate = totalBlades > 0 ? (totalBlades - visibleBlades) * 100 / totalBlades : 0;
                int totalTriangles = visibleBlades * 3 + treeCount_ * 64;

                ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "剔除效果对比：");
                ImGui::Text("总草叶数       : %d", totalBlades);
                ImGui::Text("剔除后可见数   : %d", visibleBlades);
                ImGui::Text("剔除率         : %d%%", cullRate);
                ImGui::Text("树木数量       : %d", treeCount_);
                ImGui::Text("总三角形估算   : ~%d K", totalTriangles / 1000);
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "GPU 耗时估算：");
                float grassMs = visibleBlades * 0.00002f;
                float cullMs = enableCulling_ ? totalBlades * 0.000003f : 0.0f;
                float treeMs = treeCount_ * 0.005f;
                ImGui::Text("  草地 Vertex Pass : ~%.1f ms", grassMs);
                ImGui::Text("  Compute 剔除     : ~%.2f ms", cullMs);
                ImGui::Text("  树木渲染         : ~%.1f ms", treeMs);
                ImGui::Text("  总计             : ~%.1f ms", grassMs + cullMs + treeMs);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    float windTime_ = 0.0f;
    float windStrength_ = 0.5f;
    float windAngle_ = 45.0f;
    float renderArea_ = 80.0f;
    int densityIdx_ = 1;
    int treeCount_ = 50;
    bool isSimulating_ = true;
    bool enableCulling_ = true;
    int culledVisibleCount_ = 0;

    void rebuildSimulation() {
        const int densityVals[] = {100, 1000, 10000, 100000};
        int total = densityVals[densityIdx_];
        culledVisibleCount_ = static_cast<int>(total * 0.60f);
        treeCount_ = static_cast<int>(renderArea_ * 0.5f);
    }

    void simulateCulling() {
        const int densityVals[] = {100, 1000, 10000, 100000};
        int total = densityVals[densityIdx_];
        float visRate = 0.45f + std::sin(windTime_ * 0.5f) * 0.15f;
        culledVisibleCount_ = static_cast<int>(total * visRate);
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第80章：植被渲染（草地 + 树木）\n";
    std::cout << " 高级渲染技术系列 — ch80/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch80App app;
        app.run("第80章：植被渲染");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
