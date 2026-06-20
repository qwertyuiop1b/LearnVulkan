/**
 * @file ch68_scene_ecs.cpp
 * @brief 第68章：场景图 & ECS（World / ComponentStorage / FrustumCuller）
 *
 * 【OOP 场景图 vs ECS 的对比】
 *
 *  OOP 传统做法（继承式场景图）：
 *    class Node { vector<shared_ptr<Node>> children; };
 *    class MeshNode   : public Node { Mesh* mesh; };
 *    class LightNode  : public Node { Light light; };
 *    问题：虚函数开销，cache miss（子节点指针散落在堆上）
 *
 *  ECS 做法（数据驱动）：
 *    Entity = 一个 uint32_t ID，无数据
 *    ComponentStorage<T> = 密集数组（SoA），Cache-Friendly
 *    System = 遍历特定 Component 组合的函数
 *
 * 【FrustumCuller 演示】
 *  1000 个实体，每帧 CPU 视锥裁剪，ImGui 显示：
 *    总数 / 可见数 / 裁剪比例 / 裁剪耗时
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <vulkan_tutorial/engine/scene_ecs.hpp>

#include <chrono>
#include <random>

using namespace engine;

class Ch68App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.05f, 0.10f, 0.06f};
        buildWorld();
    }

    void onUpdate() override {
        // 每帧更新相机变换并做视锥裁剪
        auto t0 = std::chrono::high_resolution_clock::now();
        Frustum frustum = makeFrustum();
        culler_.cull(world_, frustum, visible_);
        auto t1 = std::chrono::high_resolution_clock::now();
        cullMs_ = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第68章：场景图 & ECS");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 690), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Scene ECS — World / ComponentStorage / FrustumCuller", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("EcsTabs")) {

            if (ImGui::BeginTabItem("ECS 设计思想")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "ECS vs 传统继承式场景图");
                ImGui::Separator();
                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "传统 OOP 场景图（cache 不友好）");
                ImGui::TextWrapped("class Node {\n"
                                   "    vector<shared_ptr<Node>> children;  // 堆上的指针\n"
                                   "    virtual void update() = 0;          // 虚函数开销\n"
                                   "};\n"
                                   "// 遍历时：指针跳跃 → cache miss → 性能差\n\n");

                ImGui::TextColored(ImVec4(0.3f, 1, 0.4f, 1), "ECS（Cache-Friendly，SoA 存储）");
                ImGui::TextWrapped("// 实体 = 一个 ID，无方法，无数据\n"
                                   "EntityID box = world.createEntity(\"Box\");\n\n"
                                   "// 组件 = 纯数据（POD 结构体）\n"
                                   "world.add<TransformComponent>(box, {{1,0,0}, {0,0,0}, {1,1,1}});\n"
                                   "world.add<MeshComponent>(box, {vb, ib, 36});\n"
                                   "world.add<MaterialComponent>(box, {matId});\n\n"
                                   "// ComponentStorage<T> = 密集数组，连续内存\n"
                                   "// 遍历时：顺序访问，CPU prefetch 友好\n\n"
                                   "// System = 遍历特定 Component 组合的函数：\n"
                                   "world.view<MeshComponent, TransformComponent>(\n"
                                   "    [](EntityID id, MeshComponent& m, TransformComponent& t) {\n"
                                   "        renderMesh(m.vertexBuffer, t.world);\n"
                                   "    });\n");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("视锥裁剪统计")) {
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "FrustumCuller — CPU AABB 裁剪");
                ImGui::Separator();

                ImGui::Text("总实体数量   : %u", culler_.lastTotalCount());
                ImGui::Text("可见实体数   : %u", culler_.lastVisibleCount());
                float culled = culler_.lastTotalCount() > 0
                                   ? (1.0f - float(culler_.lastVisibleCount()) / culler_.lastTotalCount()) * 100.0f
                                   : 0.0f;
                ImGui::Text("裁剪比例     : %.1f %%", culled);
                ImGui::Text("裁剪耗时     : %.3f ms", cullMs_);
                ImGui::Spacing();

                // 进度条显示可见率
                float visRate =
                    culler_.lastTotalCount() > 0 ? float(culler_.lastVisibleCount()) / culler_.lastTotalCount() : 1.0f;
                ImGui::Text("可见率：");
                ImGui::ProgressBar(visRate, ImVec2(-1, 20));

                ImGui::Separator();
                ImGui::Text("实体数量设置：");
                if (ImGui::SliderInt("实体总数", &entityCount_, 100, 2000)) {
                    buildWorld();
                }
                ImGui::Checkbox("启用视锥裁剪", &enableCull_);
                ImGui::Spacing();
                ImGui::TextWrapped("// 裁剪流程：\n"
                                   "// 1. 从相机 viewProj 矩阵提取 6 个裁剪平面（Gribb-Hartmann 法）\n"
                                   "// 2. 对每个实体的 AABB 变换到世界空间\n"
                                   "// 3. AABB vs 6 平面测试（positive vertex 法）\n"
                                   "// 4. 不相交的实体不提交 draw call\n");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("ComponentStorage API")) {
                ImGui::TextColored(ImVec4(0.8f, 0.5f, 1, 1), "ComponentStorage<T> — 密集数组");
                ImGui::Separator();
                ImGui::TextWrapped("// 内部结构（SoA - Structure of Arrays）：\n"
                                   "vector<EntityID>  entities_;      // 实体 ID 列表\n"
                                   "vector<T>         components_;    // 对应的 Component 数据（连续）\n"
                                   "unordered_map<EntityID, uint32_t> indexOf_;  // ID → 数组索引\n\n"
                                   "// 添加 Component：\n"
                                   "void add(EntityID id, T component);\n\n"
                                   "// 查询 Component（O(1)）：\n"
                                   "T* get(EntityID id);\n\n"
                                   "// 遍历所有（连续内存，Cache-Friendly）：\n"
                                   "void forEach(function<void(EntityID, T&)> fn);\n\n"
                                   "// 删除（使用 swap-and-pop，O(1) 保持密集）：\n"
                                   "void remove(EntityID id);\n\n");

                ImGui::Text("当前 World 状态：");
                ImGui::Text("  实体总数            : %zu", world_.entityCount());
                ImGui::Text("  有 Transform 的实体 : %zu", world_.storage<TransformComponent>().size());
                ImGui::Text("  有 Mesh 的实体      : %zu", world_.storage<MeshComponent>().size());
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    World world_;
    FrustumCuller culler_;
    std::vector<EntityID> visible_;
    float cullMs_ = 0.0f;
    bool enableCull_ = true;
    int entityCount_ = 500;

    void buildWorld() {
        world_ = World{};
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> pos(-20.0f, 20.0f);
        std::uniform_real_distribution<float> sz(0.2f, 1.5f);

        for (int i = 0; i < entityCount_; ++i) {
            auto e = world_.createEntity();
            float x = pos(rng), y = pos(rng), z = pos(rng);
            float s = sz(rng);
            TransformComponent tc{};
            tc.position = {x, y, z};
            tc.scale = {s, s, s};
            tc.dirty = true;
            tc.world = tc.localMatrix();
            world_.add<TransformComponent>(e, tc);
            MeshComponent mc{};
            mc.aabbMin = {-0.5f * s, -0.5f * s, -0.5f * s};
            mc.aabbMax = {0.5f * s, 0.5f * s, 0.5f * s};
            world_.add<MeshComponent>(e, mc);
        }
    }

    Frustum makeFrustum() {
        glm::mat4 view = interactive_.camera().viewMatrix();
        int w = static_cast<int>(extent_.width);
        int h = static_cast<int>(extent_.height);
        float aspect = h > 0 ? float(w) / float(h) : 1.0f;
        glm::mat4 proj = interactive_.camera().projectionMatrix(aspect);
        return Frustum::fromViewProj(proj * view);
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第68章：场景图 & ECS（World / FrustumCuller）\n";
    std::cout << " 引擎封装系列 — ch68/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch68App app;
        app.run("第68章：场景图 ECS", 960, 720);
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
