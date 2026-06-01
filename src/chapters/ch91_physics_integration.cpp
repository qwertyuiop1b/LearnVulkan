/**
 * @file ch91_physics_integration.cpp
 * @brief 第91章：物理引擎集成（Bullet / Jolt 架构）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【集成架构】
 *
 *  PhysicsWorld  — 独立物理步进（fixed timestep）
 *  RigidBody       — 动态/静态/运动学刚体
 *  Collider        — Box / Sphere / Capsule / Mesh
 *  PhysicsSync     — 物理 → Transform 同步到 ECS
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

enum class BodyType { Static, Dynamic, Kinematic };
enum class ColliderShape { Box, Sphere, Capsule, Mesh };

struct RigidBodySim {
    glm::vec3 position{0.0f};
    glm::vec3 velocity{0.0f};
    glm::vec3 angularVel{0.0f};
    float rotation = 0.0f;
    float mass = 1.0f;
    float restitution = 0.5f;
    float friction = 0.4f;
    BodyType type = BodyType::Dynamic;
    ColliderShape shape = ColliderShape::Box;
    glm::vec3 halfExtents{0.5f, 0.5f, 0.5f};
    float radius = 0.5f;
    bool onGround = false;
};

class Ch91App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.05f, 0.06f, 0.09f};
        resetScene();
    }

    void onUpdate() override {
        accumulator_ += 0.016f;
        while (accumulator_ >= fixedDt_) {
            stepPhysics(fixedDt_);
            accumulator_ -= fixedDt_;
            ++physicsSteps_;
        }
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第91章：物理引擎集成");
        ImGui::Separator();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第91章：物理引擎集成（Bullet / Jolt）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("PhysicsTabs")) {
            if (ImGui::BeginTabItem("架构对比")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Bullet vs Jolt");
                ImGui::Separator();
                const char* engines[] = {"Bullet Physics", "Jolt Physics"};
                ImGui::Combo("引擎选择", &engineChoice_, engines, 2);
                ImGui::TextWrapped("共同集成模式：\n\n"
                                   "  // 初始化\n"
                                   "  physicsWorld_.init({ gravity, maxBodies, fixedDt });\n\n"
                                   "  // 每帧（固定步长）\n"
                                   "  while (accumulator >= fixedDt) {\n"
                                   "      physicsWorld_.stepSimulation(fixedDt);\n"
                                   "      syncTransformsToECS();  // pos/rot → TransformComponent\n"
                                   "      accumulator -= fixedDt;\n"
                                   "  }\n\n"
                                   "  // 渲染前\n"
                                   "  for (auto& body : dynamicBodies_)\n"
                                   "      body.transform = physicsWorld_.getTransform(body.id);");
                ImGui::Spacing();
                ImGui::Text("Bullet — 成熟稳定，文档丰富，GImpact 网格碰撞");
                ImGui::Text("Jolt   — 多线程友好，SIMD 优化，Deterministic");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("刚体模拟")) {
                if (ImGui::Button("重置场景"))
                    resetScene();
                ImGui::SameLine();
                if (ImGui::Button("发射球体"))
                    spawnSphere();
                ImGui::SliderFloat("固定步长 (s)", &fixedDt_, 0.008f, 0.033f, "%.3f");
                ImGui::SliderFloat3("重力", &gravity_.x, -20.0f, 0.0f);
                ImGui::SliderInt("子步数", &subSteps_, 1, 8);
                ImGui::Spacing();
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                ImVec2 size(420, 260);
                dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(25, 30, 40, 255));
                float groundY = pos.y + size.y - 30.0f;
                dl->AddRectFilled(
                    ImVec2(pos.x, groundY), ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(60, 70, 55, 255));
                for (auto& b : bodies_) {
                    float sx = pos.x + (b.position.x + 5.0f) / 10.0f * size.x;
                    float sy = groundY - (b.position.y + 0.5f) / 6.0f * (size.y - 40);
                    ImU32 col =
                        b.type == BodyType::Static ? IM_COL32(100, 100, 110, 255) : IM_COL32(100, 180, 255, 255);
                    if (b.shape == ColliderShape::Sphere)
                        dl->AddCircleFilled(ImVec2(sx, sy), b.radius * 12.0f, col);
                    else
                        dl->AddRectFilled(ImVec2(sx - 10, sy - 10), ImVec2(sx + 10, sy + 10), col);
                }
                ImGui::Dummy(size);
                ImGui::Text("活跃刚体 : %zu  |  物理步 : %d", bodies_.size(), physicsSteps_);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("碰撞形状")) {
                const char* shapes[] = {"Box", "Sphere", "Capsule", "Mesh (Concave)"};
                ImGui::Combo("默认形状", &defaultShape_, shapes, 4);
                ImGui::TextWrapped("Box     — btBoxShape / BoxShape\n"
                                   "Sphere  — btSphereShape / SphereShape\n"
                                   "Capsule — btCapsuleShape / CapsuleShape\n"
                                   "Mesh    — btBvhTriangleMeshShape / MeshShape\n\n"
                                   "Mesh 碰撞注意：\n"
                                   "  · 仅 Static 使用 Concave Mesh\n"
                                   "  · Dynamic 使用 Convex Hull 近似");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("ECS 同步")) {
                ImGui::TextWrapped("class PhysicsSyncSystem {\n"
                                   "    void fixedUpdate(float dt) {\n"
                                   "        world_.step(dt);\n"
                                   "        for (EntityID e : world_.view<RigidBodyComponent>()) {\n"
                                   "            auto& rb  = world_.get<RigidBodyComponent>(e);\n"
                                   "            auto& tr  = world_.get<TransformComponent>(e);\n"
                                   "            tr.position = rb.body->getPosition();\n"
                                   "            tr.rotation = rb.body->getRotation();\n"
                                   "        }\n"
                                   "    }\n"
                                   "    void prePhysics() {\n"
                                   "        // Kinematic: ECS → Physics\n"
                                   "        for (EntityID e : kinematicBodies_)\n"
                                   "            rb.body->setTransform(tr.position, tr.rotation);\n"
                                   "    }\n"
                                   "};");
                ImGui::Spacing();
                ImGui::Text("同步延迟 : %.1f ms（%d 子步）", fixedDt_ * 1000.0f * subSteps_, subSteps_);
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    std::vector<RigidBodySim> bodies_;
    glm::vec3 gravity_{0.0f, -9.8f, 0.0f};
    float fixedDt_ = 1.0f / 60.0f;
    float accumulator_ = 0.0f;
    int subSteps_ = 2;
    int physicsSteps_ = 0;
    int engineChoice_ = 1;
    int defaultShape_ = 1;

    void resetScene() {
        bodies_.clear();
        physicsSteps_ = 0;
        RigidBodySim ground{};
        ground.position = {0, -0.5f, 0};
        ground.type = BodyType::Static;
        ground.shape = ColliderShape::Box;
        ground.halfExtents = {5.0f, 0.5f, 5.0f};
        bodies_.push_back(ground);
        for (int i = 0; i < 3; ++i) {
            RigidBodySim box{};
            box.position = {float(i - 1) * 1.2f, 2.0f + i * 0.8f, 0};
            box.mass = 1.0f + i * 0.5f;
            box.restitution = 0.3f + i * 0.1f;
            box.shape = ColliderShape::Box;
            bodies_.push_back(box);
        }
    }

    void spawnSphere() {
        RigidBodySim s{};
        s.position = {0, 5.0f, 0};
        s.velocity = {2.0f, 0.0f, 0.0f};
        s.shape = ColliderShape::Sphere;
        s.radius = 0.4f;
        s.restitution = 0.7f;
        bodies_.push_back(s);
    }

    void stepPhysics(float dt) {
        for (auto& b : bodies_) {
            if (b.type != BodyType::Dynamic)
                continue;
            b.velocity += gravity_ * dt;
            b.position += b.velocity * dt;
            b.rotation += b.angularVel.y * dt;
            if (b.position.y - b.radius < 0.0f) {
                b.position.y = b.radius;
                b.velocity.y = -b.velocity.y * b.restitution;
                b.velocity.x *= (1.0f - b.friction);
                b.onGround = std::abs(b.velocity.y) < 0.1f;
            }
        }
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第91章：物理引擎集成\n";
    std::cout << " 游戏引擎系列 — ch91/4\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch91App app;
        app.run("第91章：物理引擎集成");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
