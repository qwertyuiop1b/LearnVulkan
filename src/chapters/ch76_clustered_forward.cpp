/**
 * @file ch76_clustered_forward.cpp
 * @brief 第76章：聚簇前向着色（Clustered Forward Shading）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【为什么需要 Clustered Shading】
 *
 *  传统前向着色（Forward Shading）：
 *    · 每个片元 × 每个光源 = O(N×M) 计算
 *    · 100个光源 × 100万片元 = 1亿次光照计算
 *    · 性能极差
 *
 *  Tiled Deferred / Tiled Forward：
 *    · 将屏幕分成 2D 格子（16×16 px 每格）
 *    · 每格存储影响它的光源列表
 *    · O(N × avg_lights_per_tile) 但深度方向无优化
 *
 *  Clustered Forward（聚簇前向）：
 *    · 3D 格子（深度也分割，对数分布）
 *    · 每个 cluster 的深度范围很窄，光源很少
 *    · 解决了深度方向的穿透问题
 *
 * 【格子划分】
 *  · 屏幕 16×9 = 144 个 tile（16px × 16px 每格，1920/16 × 1080/16）
 *  · 深度 24 层（对数分布 near~far）
 *  · 共 144 × 24 = 3456 个 cluster
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <array>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

static constexpr int CLUSTER_X      = 16;
static constexpr int CLUSTER_Y      = 9;
static constexpr int CLUSTER_Z      = 24;
static constexpr int TOTAL_CLUSTERS = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;

class Ch76App : public DemoApp {
protected:
    void onInit() override
    {
        bgColor_ = {0.05f, 0.06f, 0.10f};
        rebuildLightStats();
    }

    void onUpdate() override
    {
        frameTime_ += 0.016f;
        if (isAnimating_) {
            rebuildLightStats();
        }
    }

    void buildUi() override
    {
        interactive_.buildDebugPanel("第76章：聚簇前向着色");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第76章：聚簇前向着色（Clustered Forward Shading）", nullptr))
        { ImGui::End(); return; }

        if (ImGui::BeginTabBar("ClusteredTabs")) {

            // ── Tab 1: 技术对比 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("技术对比")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "多光源着色技术演进");
                ImGui::Separator();

                struct TechEntry {
                    const char* name;
                    const char* complexity;
                    const char* depthHandling;
                    const char* maxLights;
                    const char* notes;
                };
                std::array<TechEntry, 4> techs = {{
                    {"传统前向",     "O(N×M)",    "无",      "8-16",    "最简单，光源极少"},
                    {"延迟着色",     "O(N+M)",    "无",      "1000+",   "不支持透明，MSAA 难"},
                    {"Tiled Forward","O(N×T)",    "无分割",  "100-200", "深度穿透问题"},
                    {"Clustered",    "O(N×C)",    "对数分割","200-1000","现代主流方案"},
                }};

                ImGui::Columns(5, "techCols");
                ImGui::Text("技术");       ImGui::NextColumn();
                ImGui::Text("复杂度");     ImGui::NextColumn();
                ImGui::Text("深度处理");   ImGui::NextColumn();
                ImGui::Text("推荐光源数"); ImGui::NextColumn();
                ImGui::Text("备注");       ImGui::NextColumn();
                ImGui::Separator();
                for (int ti = 0; ti < static_cast<int>(techs.size()); ++ti) {
                    auto& t = techs[ti];
                    bool highlight = (ti == 3);
                    if (highlight) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f,1,0.5f,1));
                    ImGui::Text("%s", t.name);         ImGui::NextColumn();
                    ImGui::Text("%s", t.complexity);   ImGui::NextColumn();
                    ImGui::Text("%s", t.depthHandling);ImGui::NextColumn();
                    ImGui::Text("%s", t.maxLights);    ImGui::NextColumn();
                    ImGui::Text("%s", t.notes);        ImGui::NextColumn();
                    if (highlight) ImGui::PopStyleColor();
                }
                ImGui::Columns(1);
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.4f,0.8f,1,1), "Cluster 索引计算：");
                ImGui::TextWrapped(
                    "// 从片元坐标计算所属 cluster：\n"
                    "uint clusterX = uint(gl_FragCoord.x / tileSize.x);\n"
                    "uint clusterY = uint(gl_FragCoord.y / tileSize.y);\n\n"
                    "// 深度层（对数分布，near/far 间均匀分割 log 空间）：\n"
                    "float logFarNear = log2(zFar / zNear);\n"
                    "uint clusterZ = uint(log2(linearDepth / zNear)\n"
                    "                * NUM_DEPTH_SLICES / logFarNear);\n\n"
                    "uint clusterIdx = clusterX\n"
                    "    + clusterY * CLUSTER_X\n"
                    "    + clusterZ * CLUSTER_X * CLUSTER_Y;\n\n"
                    "// 查找光源列表：\n"
                    "uint lightStart = clusterLightList[clusterIdx].offset;\n"
                    "uint lightCount = clusterLightList[clusterIdx].count;\n"
                    "for (uint i = 0; i < lightCount; ++i) {\n"
                    "    uint lightIdx = lightIndices[lightStart + i];\n"
                    "    // 计算该光源的贡献\n"
                    "    result += calcLight(lights[lightIdx]);\n"
                    "}");
                ImGui::EndTabItem();
            }

            // ── Tab 2: 格子划分 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("格子划分")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Cluster 3D 格子参数");
                ImGui::Separator();

                ImGui::Text("屏幕分辨率   : 1920 × 1080");
                ImGui::Text("Tile 大小    : 16 × 16 px");
                ImGui::Text("屏幕 Tile 数 : %d × %d = %d", CLUSTER_X, CLUSTER_Y, CLUSTER_X * CLUSTER_Y);
                ImGui::Text("深度层数     : %d 层（对数分布）", CLUSTER_Z);
                ImGui::Text("总 Cluster 数: %d", TOTAL_CLUSTERS);
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.4f,0.8f,1,1), "深度层分布（对数，Near=0.1m, Far=1000m）：");
                ImGui::Columns(3, "depthCols");
                ImGui::Text("层级"); ImGui::NextColumn();
                ImGui::Text("近边界"); ImGui::NextColumn();
                ImGui::Text("远边界"); ImGui::NextColumn();
                ImGui::Separator();
                const float zNear = 0.1f, zFar = 1000.0f;
                for (int z = 0; z < CLUSTER_Z; z += 4) {
                    float near = zNear * std::pow(zFar/zNear, float(z) / CLUSTER_Z);
                    float far  = zNear * std::pow(zFar/zNear, float(z+1) / CLUSTER_Z);
                    ImGui::Text("Slice %2d", z);  ImGui::NextColumn();
                    ImGui::Text("%.1f m", near);   ImGui::NextColumn();
                    ImGui::Text("%.1f m", far);    ImGui::NextColumn();
                }
                ImGui::Text("...");  ImGui::NextColumn();
                ImGui::Text("...");  ImGui::NextColumn();
                ImGui::Text("...");  ImGui::NextColumn();
                ImGui::Columns(1);
                ImGui::EndTabItem();
            }

            // ── Tab 3: Compute 分配 ────────────────────────────────────────
            if (ImGui::BeginTabItem("Compute 光源分配")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Compute Shader 光源分配算法");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "// Compute Shader：cluster_assign.comp\n"
                    "// 每个工作组处理一个 cluster\n"
                    "layout(local_size_x=1, local_size_y=1, local_size_z=1) in;\n\n"
                    "void main() {\n"
                    "    uint idx = gl_GlobalInvocationID.x;\n"
                    "    if (idx >= TOTAL_CLUSTERS) return;\n\n"
                    "    // 从 cluster 索引计算 3D 位置\n"
                    "    uint z = idx / (CLUSTER_X * CLUSTER_Y);\n"
                    "    uint y = (idx %% (CLUSTER_X * CLUSTER_Y)) / CLUSTER_X;\n"
                    "    uint x = idx %% CLUSTER_X;\n\n"
                    "    // 计算 cluster 的 AABB（视锥体子空间）\n"
                    "    AABB clusterAABB = computeClusterAABB(x, y, z);\n\n"
                    "    // 测试每个点光源的球 vs cluster AABB\n"
                    "    uint lightCount = 0;\n"
                    "    for (uint i = 0; i < numLights; ++i) {\n"
                    "        if (sphereAABBIntersect(lights[i].pos, lights[i].radius,\n"
                    "                               clusterAABB)) {\n"
                    "            lightList[idx * MAX_LIGHTS + lightCount++] = i;\n"
                    "        }\n"
                    "    }\n"
                    "    lightCounts[idx] = lightCount;\n"
                    "}");
                ImGui::EndTabItem();
            }

            // ── Tab 4: 实时统计 ────────────────────────────────────────────
            if (ImGui::BeginTabItem("实时统计")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "光源分配统计");
                ImGui::Separator();

                bool changed = false;
                changed |= ImGui::SliderInt("总光源数", &numLights_, 10, 200);
                ImGui::Checkbox("动态更新", &isAnimating_);
                if (changed) rebuildLightStats();
                ImGui::Spacing();

                ImGui::Text("总光源数          : %d",    numLights_);
                ImGui::Text("活跃 Cluster 数   : %d / %d",   activeClusters_, TOTAL_CLUSTERS);
                ImGui::Text("每 Cluster 平均   : %.1f 光源", avgLightsPerCluster_);
                ImGui::Text("最大光源数 Cluster: %d 光源",   maxLightsInCluster_);
                ImGui::Text("空 Cluster 数     : %d",        emptyClusters_);
                ImGui::Text("Compute 耗时估算  : ~%.2f ms",
                    numLights_ * TOTAL_CLUSTERS * 0.000003f);
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.4f,0.8f,1,1), "光源分布直方图（每Cluster光源数）：");
                ImGui::Text("0 光源：");
                ImGui::SameLine();
                ImGui::ProgressBar(float(emptyClusters_) / TOTAL_CLUSTERS,
                    ImVec2(300, 14));

                ImGui::Text("1-2光源：");
                ImGui::SameLine();
                ImGui::ProgressBar(float(lowLightClusters_) / TOTAL_CLUSTERS,
                    ImVec2(300, 14));

                ImGui::Text("3-8光源：");
                ImGui::SameLine();
                ImGui::ProgressBar(float(midLightClusters_) / TOTAL_CLUSTERS,
                    ImVec2(300, 14));

                ImGui::Text("8+光源：");
                ImGui::SameLine();
                ImGui::ProgressBar(float(highLightClusters_) / TOTAL_CLUSTERS,
                    ImVec2(300, 14));
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

private:
    int   numLights_          = 50;
    bool  isAnimating_        = false;
    float frameTime_          = 0.0f;
    int   activeClusters_     = 0;
    float avgLightsPerCluster_ = 0.0f;
    int   maxLightsInCluster_ = 0;
    int   emptyClusters_      = 0;
    int   lowLightClusters_   = 0;
    int   midLightClusters_   = 0;
    int   highLightClusters_  = 0;

    void rebuildLightStats()
    {
        std::mt19937 rng(static_cast<uint32_t>(frameTime_ * 10));
        std::uniform_int_distribution<int> clusterDist(0, TOTAL_CLUSTERS - 1);
        std::uniform_real_distribution<float> radiusDist(0.1f, 0.4f);

        std::vector<int> clusterLightCount(TOTAL_CLUSTERS, 0);
        for (int i = 0; i < numLights_; ++i) {
            float radius    = radiusDist(rng);
            int   numHits   = static_cast<int>(radius * TOTAL_CLUSTERS * 0.02f) + 1;
            for (int h = 0; h < numHits; ++h) {
                int c = clusterDist(rng);
                clusterLightCount[c]++;
            }
        }

        activeClusters_    = 0;
        maxLightsInCluster_ = 0;
        emptyClusters_     = 0;
        lowLightClusters_  = 0;
        midLightClusters_  = 0;
        highLightClusters_ = 0;
        int totalAssigned  = 0;

        for (int c : clusterLightCount) {
            if (c == 0) { ++emptyClusters_; continue; }
            ++activeClusters_;
            totalAssigned += c;
            maxLightsInCluster_ = std::max(maxLightsInCluster_, c);
            if (c <= 2)      ++lowLightClusters_;
            else if (c <= 8) ++midLightClusters_;
            else             ++highLightClusters_;
        }
        avgLightsPerCluster_ = activeClusters_ > 0
            ? float(totalAssigned) / activeClusters_ : 0.0f;
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第76章：聚簇前向着色（Clustered Forward Shading）\n";
    std::cout << " 高级渲染技术系列 — ch76/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch76App app;
        app.run("第76章：聚簇前向着色");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
