/**
 * @file ch79_terrain_lod.cpp
 * @brief 第79章：地形 LOD（四叉树 + GPU 曲面细分）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【四叉树 LOD】
 *
 *  地形被划分为 Patch（面片），每个 Patch 可以继续细分为 4 个子 Patch，
 *  构成四叉树（Quadtree）结构。
 *
 *  LOD 判断标准：
 *    distance(camera, patchCenter) / patchSize > lodThreshold
 *    · 距离大 → 不细分（低精度，合并 Patch）
 *    · 距离小 → 细分（高精度，更多 Patch）
 *
 * 【GPU 曲面细分（Tessellation）】
 *
 *  Vulkan 使用三个着色器阶段：
 *  1. Tessellation Control Shader (TCS / Hull Shader)：
 *     · 决定细分级别（gl_TessLevelOuter / gl_TessLevelInner）
 *     · 根据距离设置 TessLevel：近处高，远处低
 *  2. 固定功能细分器（Tessellator）
 *  3. Tessellation Evaluation Shader (TES / Domain Shader)：
 *     · 在细分后的顶点上采样高度图
 *     · y = heightmap.sample(uv) × heightScale
 *
 * 【程序化高度图（分形噪声）】
 *
 *  h(x, z) = Σ(i=0..2) amplitude_i × sin(x×freq_i) × cos(z×freq_i)
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

static constexpr int TERRAIN_GRID = 8;
static constexpr int MAX_LOD = 4;
static constexpr float PATCH_SIZE = 64.0f;
static constexpr float TERRAIN_SIZE = PATCH_SIZE * TERRAIN_GRID;

struct TerrainPatch {
    glm::vec2 center;
    float size;
    int lodLevel;
    int tessLevel;
    int vertexCount;
};

class Ch79App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.08f, 0.12f, 0.06f};
        rebuildPatches();
    }

    void onUpdate() override {
        if (isAnimating_) {
            cameraAngle_ += 0.3f * 0.016f;
            float r = TERRAIN_SIZE * 0.4f;
            cameraPos_.x = std::cos(cameraAngle_) * r;
            cameraPos_.z = std::sin(cameraAngle_) * r;
            cameraPos_.y = cameraHeight_;
        }
        rebuildPatches();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第79章：地形 LOD");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第79章：地形 LOD（四叉树 + 曲面细分）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("TerrainTabs")) {

            // ── Tab 1: 四叉树 LOD ──────────────────────────────────────────
            if (ImGui::BeginTabItem("四叉树 LOD")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "四叉树地形 LOD");
                ImGui::Separator();
                ImGui::TextWrapped("地形划分方式：\n"
                                   "  · 根节点 = 整个地形（512m × 512m）\n"
                                   "  · 每次细分 → 4 个子 Patch（各边长/2）\n"
                                   "  · 最大细分 4 层（最小 Patch = 32m × 32m）\n\n"
                                   "LOD 判断（递归细分）：\n"
                                   "  bool shouldSplit(Patch p) {\n"
                                   "      float dist = distance(camera, p.center);\n"
                                   "      return dist < p.size * lodFactor;  // 距离小则细分\n"
                                   "  }\n\n"
                                   "  // 伪代码：递归构建可见 Patch 列表\n"
                                   "  void traverse(Patch p, vector<Patch>& visible) {\n"
                                   "      if (!frustumContains(p)) return;  // 视锥剔除\n"
                                   "      if (p.level >= MAX_LOD || !shouldSplit(p)) {\n"
                                   "          visible.push_back(p);  // 叶节点，加入渲染列表\n"
                                   "          return;\n"
                                   "      }\n"
                                   "      for (auto& child : p.children())  // 细分4个子节点\n"
                                   "          traverse(child, visible);\n"
                                   "  }\n\n"
                                   "T-Junction 问题：\n"
                                   "  · 相邻 Patch LOD 不同时，边界会出现裂缝\n"
                                   "  · 解决：根据邻居 LOD 调整边界 TessLevel（Skirt）");
                ImGui::EndTabItem();
            }

            // ── Tab 2: GPU 曲面细分 ────────────────────────────────────────
            if (ImGui::BeginTabItem("GPU 曲面细分")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Tessellation Control / Eval Shader");
                ImGui::Separator();
                ImGui::TextWrapped("TCS（Tessellation Control Shader）：\n\n"
                                   "  // 输入：patch 的 4 个控制点\n"
                                   "  layout(vertices = 4) out;\n\n"
                                   "  void main() {\n"
                                   "      // 根据相机距离动态设置细分级别\n"
                                   "      float dist = distance(camera, patchCenter);\n"
                                   "      float t = clamp(1.0 - dist / maxDist, 0.0, 1.0);\n"
                                   "      float tessLevel = mix(minTess, maxTess, t);\n\n"
                                   "      // 外边细分（影响边界）\n"
                                   "      gl_TessLevelOuter[0] = tessLevel;\n"
                                   "      gl_TessLevelOuter[1] = tessLevel;\n"
                                   "      gl_TessLevelOuter[2] = tessLevel;\n"
                                   "      gl_TessLevelOuter[3] = tessLevel;\n"
                                   "      // 内部细分\n"
                                   "      gl_TessLevelInner[0] = tessLevel;\n"
                                   "      gl_TessLevelInner[1] = tessLevel;\n"
                                   "  }\n\n"
                                   "TES（Tessellation Evaluation Shader）：\n\n"
                                   "  layout(quads, equal_spacing, ccw) in;\n\n"
                                   "  void main() {\n"
                                   "      // gl_TessCoord.xy = [0,1]² 细分坐标\n"
                                   "      vec2 uv = mix(patch[0].uv, patch[3].uv, gl_TessCoord.xy);\n\n"
                                   "      // 从高度图采样（程序化或纹理）\n"
                                   "      float h = texture(heightmap, uv).r * heightScale;\n\n"
                                   "      // 输出世界坐标\n"
                                   "      vec3 pos = vec3(uv.x * terrainSize, h, uv.y * terrainSize);\n"
                                   "      gl_Position = viewProj * vec4(pos, 1.0);\n"
                                   "  }");
                ImGui::EndTabItem();
            }

            // ── Tab 3: 程序化高度图 ────────────────────────────────────────
            if (ImGui::BeginTabItem("程序化高度图")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "分形噪声高度图");
                ImGui::Separator();
                ImGui::TextWrapped("程序化高度公式（3 层 Octave 叠加）：\n\n"
                                   "  float sampleHeight(float x, float z) {\n"
                                   "      float h = 0.0;\n"
                                   "      // 低频层（大地形起伏）\n"
                                   "      h += sin(x*0.01) * cos(z*0.013) * 30.0;\n"
                                   "      // 中频层（丘陵）\n"
                                   "      h += sin(x*0.04 + z*0.035) * cos(z*0.04) * 12.0;\n"
                                   "      // 高频层（细节）\n"
                                   "      h += sin(x*0.15 + z*0.12) * sin(z*0.18) * 4.0;\n"
                                   "      return h + 20.0;  // 基础高度\n"
                                   "  }\n\n"
                                   "特点：\n"
                                   "  · 无需纹理文件，运行时生成\n"
                                   "  · 可无缝平铺（周期函数）\n"
                                   "  · 可在 CPU（四叉树构建）和 GPU（TES）中使用相同公式");

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "高度采样预览：");
                ImGui::Text("位置 (0, 0)      → h = %.1f m", sampleHeight(0.0f, 0.0f));
                ImGui::Text("位置 (100, 50)   → h = %.1f m", sampleHeight(100.0f, 50.0f));
                ImGui::Text("位置 (256, 256)  → h = %.1f m", sampleHeight(256.0f, 256.0f));
                ImGui::Text("最小高度         : ~%.0f m", -30.0f - 12.0f - 4.0f + 20.0f);
                ImGui::Text("最大高度         : ~%.0f m", 30.0f + 12.0f + 4.0f + 20.0f);
                ImGui::EndTabItem();
            }

            // ── Tab 4: 统计展示 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("统计展示")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "实时 LOD 统计");
                ImGui::Separator();

                ImGui::Checkbox("动态相机", &isAnimating_);
                ImGui::SliderFloat("相机高度 (m)", &cameraHeight_, 10.0f, 500.0f, "%.0f");
                ImGui::SliderFloat("LOD 系数", &lodFactor_, 0.5f, 5.0f, "%.1f");
                ImGui::SliderInt("最大 TessLevel", &maxTessLevel_, 2, 64);
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "当前帧统计：");
                int totalPatch = static_cast<int>(patches_.size());
                int totalVerts = 0;
                std::array<int, 5> tessDistrib = {0, 0, 0, 0, 0};
                for (auto& p : patches_) {
                    totalVerts += p.vertexCount;
                    int idx = std::min(4, p.tessLevel / 16);
                    ++tessDistrib[idx];
                }
                ImGui::Text("可见 Patch 数     : %d", totalPatch);
                ImGui::Text("顶点总数估算      : ~%d K", totalVerts / 1000);
                ImGui::Text("相机位置          : (%.0f, %.0f, %.0f)", cameraPos_.x, cameraPos_.y, cameraPos_.z);
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "TessLevel 分布：");
                const char* tessLabels[] = {"2-16", "17-32", "33-48", "49-64", "64+"};
                for (int i = 0; i < 5; ++i) {
                    ImGui::Text("  TessLevel %s:", tessLabels[i]);
                    ImGui::SameLine(190);
                    float frac = totalPatch > 0 ? float(tessDistrib[i]) / totalPatch : 0;
                    ImGui::ProgressBar(frac, ImVec2(200, 14));
                    ImGui::SameLine();
                    ImGui::Text("%d", tessDistrib[i]);
                }
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    std::vector<TerrainPatch> patches_;
    glm::vec3 cameraPos_ = {0.0f, 100.0f, 0.0f};
    float cameraAngle_ = 0.0f;
    float cameraHeight_ = 120.0f;
    float lodFactor_ = 2.0f;
    int maxTessLevel_ = 32;
    bool isAnimating_ = true;

    static float sampleHeight(float x, float z) {
        float h = 0.0f;
        h += std::sin(x * 0.01f) * std::cos(z * 0.013f) * 30.0f;
        h += std::sin(x * 0.04f + z * 0.035f) * std::cos(z * 0.04f) * 12.0f;
        h += std::sin(x * 0.15f + z * 0.12f) * std::sin(z * 0.18f) * 4.0f;
        return h + 20.0f;
    }

    int computeTessLevel(float distance) const {
        float t = std::max(0.0f, 1.0f - distance / (TERRAIN_SIZE * 0.5f));
        int level = static_cast<int>(2.0f + t * t * (maxTessLevel_ - 2));
        return std::clamp(level, 2, maxTessLevel_);
    }

    void rebuildPatches() {
        patches_.clear();
        for (int gz = 0; gz < TERRAIN_GRID; ++gz) {
            for (int gx = 0; gx < TERRAIN_GRID; ++gx) {
                glm::vec2 center = {
                    (gx + 0.5f) * PATCH_SIZE - TERRAIN_SIZE * 0.5f,
                    (gz + 0.5f) * PATCH_SIZE - TERRAIN_SIZE * 0.5f,
                };
                float dist = glm::length(glm::vec2(cameraPos_.x, cameraPos_.z) - center);
                int tess = computeTessLevel(dist);
                int verts = (tess + 1) * (tess + 1);
                patches_.push_back({center, PATCH_SIZE, 0, tess, verts});
            }
        }
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第79章：地形 LOD（四叉树 + 曲面细分）\n";
    std::cout << " 高级渲染技术系列 — ch79/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch79App app;
        app.run("第79章：地形 LOD");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
