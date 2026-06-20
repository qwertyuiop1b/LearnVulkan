/**
 * @file ch89_navmesh_pathfinding.cpp
 * @brief 第89章：导航网格与寻路（NavMesh + A*）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【流程】
 *
 *  1. NavMesh 烘焙 — 从关卡几何体生成可行走多边形
 *  2. A* 寻路     — 在多边形邻接图上搜索最短路径
 *  3. 路径平滑   — Funnel Algorithm / Catmull-Rom 样条
 *  4. 动态避障   — RVO / 局部力场修正
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <array>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include <iostream>

struct GridCell {
    bool walkable = true;
    float cost = 1.0f;
};

struct PathNode {
    int x, y;
    float g, f;
    int parentX, parentY;
    bool operator>(const PathNode& o) const {
        return f > o.f;
    }
};

class Ch89App : public DemoApp {
  protected:
    static constexpr int GRID_W = 24;
    static constexpr int GRID_H = 18;

    void onInit() override {
        bgColor_ = {0.05f, 0.06f, 0.08f};
        initGrid();
    }

    void onUpdate() override {
        if (agentMoving_ && !path_.empty()) {
            moveTimer_ += 0.016f;
            if (moveTimer_ > 0.15f) {
                moveTimer_ = 0.0f;
                if (pathIndex_ < static_cast<int>(path_.size())) {
                    agentPos_ = path_[pathIndex_++];
                } else {
                    agentMoving_ = false;
                }
            }
        }
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第89章：NavMesh 寻路");
        ImGui::Separator();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第89章：导航网格与寻路（A*）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("NavTabs")) {
            if (ImGui::BeginTabItem("网格编辑")) {
                ImGui::Text("左键：切换障碍  |  右键：设置起点/终点");
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                float cellSize = 16.0f;
                ImVec2 gridSize(GRID_W * cellSize, GRID_H * cellSize);
                for (int y = 0; y < GRID_H; ++y) {
                    for (int x = 0; x < GRID_W; ++x) {
                        ImVec2 c0(pos.x + x * cellSize, pos.y + y * cellSize);
                        ImVec2 c1(c0.x + cellSize, c0.y + cellSize);
                        ImU32 col = grid_[y][x].walkable ? IM_COL32(40, 55, 40, 255) : IM_COL32(80, 40, 40, 255);
                        dl->AddRectFilled(c0, c1, col);
                        dl->AddRect(c0, c1, IM_COL32(60, 60, 60, 255));
                    }
                }
                for (size_t i = 0; i < path_.size(); ++i) {
                    auto& p = path_[i];
                    ImVec2 c(pos.x + p.x * cellSize + 2, pos.y + p.y * cellSize + 2);
                    dl->AddRectFilled(c, ImVec2(c.x + cellSize - 4, c.y + cellSize - 4), IM_COL32(255, 200, 50, 180));
                }
                ImVec2 startPos(pos.x + start_.x * cellSize + 3, pos.y + start_.y * cellSize + 3);
                dl->AddCircleFilled(startPos, 5.0f, IM_COL32(50, 255, 80, 255));
                ImVec2 endPos(pos.x + end_.x * cellSize + cellSize * 0.5f, pos.y + end_.y * cellSize + cellSize * 0.5f);
                dl->AddCircleFilled(endPos, 5.0f, IM_COL32(255, 80, 80, 255));
                ImVec2 agentDraw(pos.x + agentPos_.x * cellSize + cellSize * 0.5f,
                                 pos.y + agentPos_.y * cellSize + cellSize * 0.5f);
                dl->AddCircleFilled(agentDraw, 4.0f, IM_COL32(100, 180, 255, 255));
                ImGui::InvisibleButton("grid", gridSize);
                if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
                    ImVec2 mp = ImGui::GetMousePos();
                    int gx = int((mp.x - pos.x) / cellSize);
                    int gy = int((mp.y - pos.y) / cellSize);
                    if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H)
                        grid_[gy][gx].walkable = !grid_[gy][gx].walkable;
                }
                if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(1)) {
                    ImVec2 mp = ImGui::GetMousePos();
                    int gx = int((mp.x - pos.x) / cellSize);
                    int gy = int((mp.y - pos.y) / cellSize);
                    if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H) {
                        if (ImGui::GetIO().KeyShift)
                            end_ = {gx, gy};
                        else
                            start_ = {gx, gy};
                    }
                }
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("A* 寻路")) {
                if (ImGui::Button("运行 A* 寻路"))
                    runAStar();
                ImGui::SameLine();
                if (ImGui::Button("开始移动 Agent")) {
                    pathIndex_ = 0;
                    agentPos_ = start_;
                    agentMoving_ = !path_.empty();
                }
                ImGui::Spacing();
                ImGui::Text("起点 : (%d, %d)  终点 : (%d, %d)", start_.x, start_.y, end_.x, end_.y);
                ImGui::Text("路径长度 : %zu 格", path_.size());
                ImGui::Text("探索节点 : %d", nodesExplored_);
                ImGui::Text("耗时     : %.3f ms", searchTimeMs_);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("NavMesh 烘焙")) {
                ImGui::TextWrapped("Recast Navigation 烘焙流程：\n\n"
                                   "  1. Voxelize — 将三角网格体素化\n"
                                   "  2. Filter   — 去除不可行走区域\n"
                                   "  3. Region   — 分水岭分区\n"
                                   "  4. Contour  — 提取轮廓多边形\n"
                                   "  5. PolyMesh — 生成凸多边形 NavMesh\n\n"
                                   "  dtNavMeshQuery::findPath(startPoly, endPoly)\n"
                                   "  → 返回多边形序列 + 路径点");
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    std::array<std::array<GridCell, GRID_W>, GRID_H> grid_{};
    glm::ivec2 start_{2, 2};
    glm::ivec2 end_{20, 15};
    glm::ivec2 agentPos_{2, 2};
    std::vector<glm::ivec2> path_;
    int pathIndex_ = 0;
    bool agentMoving_ = false;
    float moveTimer_ = 0.0f;
    int nodesExplored_ = 0;
    float searchTimeMs_ = 0.0f;

    void initGrid() {
        for (int y = 0; y < GRID_H; ++y)
            for (int x = 0; x < GRID_W; ++x)
                grid_[y][x] = {true, 1.0f};
        for (int x = 8; x < 16; ++x)
            grid_[6][x].walkable = false;
        for (int y = 4; y < 12; ++y)
            grid_[y][14].walkable = false;
    }

    void runAStar() {
        auto t0 = std::chrono::steady_clock::now();
        path_.clear();
        nodesExplored_ = 0;
        std::priority_queue<PathNode, std::vector<PathNode>, std::greater<PathNode>> open;
        std::array<std::array<bool, GRID_W>, GRID_H> closed{};
        std::array<std::array<float, GRID_W>, GRID_H> gScore{};
        std::array<std::array<glm::ivec2, GRID_W>, GRID_H> cameFrom{};
        for (auto& row : gScore)
            row.fill(1e9f);
        auto heuristic = [&](int x, int y) { return float(std::abs(x - end_.x) + std::abs(y - end_.y)); };
        gScore[start_.y][start_.x] = 0;
        open.push({start_.x, start_.y, 0, heuristic(start_.x, start_.y), -1, -1});
        const int dx[] = {0, 1, 0, -1, 1, 1, -1, -1};
        const int dy[] = {-1, 0, 1, 0, -1, 1, 1, -1};
        bool found = false;
        while (!open.empty()) {
            PathNode cur = open.top();
            open.pop();
            if (closed[cur.y][cur.x])
                continue;
            closed[cur.y][cur.x] = true;
            ++nodesExplored_;
            if (cur.x == end_.x && cur.y == end_.y) {
                found = true;
                glm::ivec2 p = end_;
                while (p.x != start_.x || p.y != start_.y) {
                    path_.push_back(p);
                    p = cameFrom[p.y][p.x];
                }
                path_.push_back(start_);
                std::reverse(path_.begin(), path_.end());
                break;
            }
            for (int d = 0; d < 8; ++d) {
                int nx = cur.x + dx[d];
                int ny = cur.y + dy[d];
                if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H)
                    continue;
                if (!grid_[ny][nx].walkable || closed[ny][nx])
                    continue;
                float stepCost = (d < 4) ? 1.0f : 1.414f;
                float ng = cur.g + stepCost;
                if (ng < gScore[ny][nx]) {
                    gScore[ny][nx] = ng;
                    cameFrom[ny][nx] = {cur.x, cur.y};
                    open.push({nx, ny, ng, ng + heuristic(nx, ny), cur.x, cur.y});
                }
            }
        }
        if (!found)
            path_.clear();
        auto t1 = std::chrono::steady_clock::now();
        searchTimeMs_ = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第89章：导航网格与寻路\n";
    std::cout << " 游戏 AI 系列 — ch89/6\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch89App app;
        app.run("第89章：NavMesh 寻路");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
