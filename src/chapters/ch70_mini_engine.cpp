/**
 * @file ch70_mini_engine.cpp
 * @brief 第70章：迷你引擎整合（MiniEngine / Application 框架）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【本章是整个系列的高潮】
 *
 *  对比 ch60_outdoor_demo.cpp（~1600 行裸 Vulkan）：
 *  使用 MiniEngine 框架重写同等场景：
 *    - 设备初始化 (ch61)：5 行
 *    - 缓冲区 (ch62)：每个 1-2 行
 *    - 纹理 (ch63)：cache.load("tex.png") 1 行
 *    - 着色器 (ch64)：lib.registerProgram(...) 1 行
 *    - 管线 (ch65)：builder.setXxx().build() 5 行
 *    - 描述符 (ch66)：builder.bindXxx().build() 3 行
 *    - 录制 (ch67)：DrawCallBatch + RAII scope
 *    - ECS (ch68)：world.view<A,B>([]{...}) 1 行
 *    - 材质 (ch69)：inst->set("albedo", ...) 1 行
 *
 * 【Application 框架】
 *  class MyGame : public Application {
 *      void onInit(MiniEngine& eng) override { ... 建场景 ... }
 *      void onUpdate(MiniEngine& eng, float dt) override { ... 逻辑 ... }
 *      void onRender(MiniEngine& eng, FrameContext& ctx) override { ... draw ... }
 *  };
 *  MiniEngine engine;
 *  engine.run(config, MyGame{});
 *
 * 【本 Demo】
 *  ImGui 面板展示：
 *  - 引擎各子系统状态
 *  - ch60 vs ch70 代码量对比（可视化图表）
 *  - 引擎调用关系架构图
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>

// ch61–ch69 所有封装层的头文件
#include <vulkan_tutorial/engine/rhi_device.hpp>
#include <vulkan_tutorial/engine/rhi_buffer.hpp>
#include <vulkan_tutorial/engine/rhi_texture.hpp>
#include <vulkan_tutorial/engine/rhi_shader.hpp>
#include <vulkan_tutorial/engine/pipeline_builder.hpp>
#include <vulkan_tutorial/engine/descriptor_manager.hpp>
#include <vulkan_tutorial/engine/command_recorder.hpp>
#include <vulkan_tutorial/engine/scene_ecs.hpp>
#include <vulkan_tutorial/engine/material_system.hpp>

#include <algorithm>
#include <random>
#include <sstream>

// ─── 代码量统计（各章节对比）────────────────────────────────────────────────

struct ChapterCodeCount {
    const char* name;
    int rawLines;    // 裸 Vulkan（ch60 风格）
    int engineLines; // 使用引擎封装后
    const char* module;
};

static const ChapterCodeCount CODE_COUNTS[] = {
    {"设备初始化", 180, 5, "ch61 RHIDevice"},
    {"缓冲区管理", 80, 3, "ch62 Buffer"},
    {"纹理加载", 90, 2, "ch63 TextureCache"},
    {"着色器+Layout", 60, 4, "ch64 ShaderProgram"},
    {"管线创建", 80, 6, "ch65 PipelineBuilder"},
    {"描述符集", 40, 3, "ch66 DescriptorBuilder"},
    {"命令录制", 30, 2, "ch67 CommandRecorder"},
    {"场景管理", 120, 8, "ch68 World/ECS"},
    {"材质参数", 50, 4, "ch69 MaterialInstance"},
    {"主循环+同步", 150, 20, "框架"},
};
static const int N_MODULES = static_cast<int>(sizeof(CODE_COUNTS) / sizeof(CODE_COUNTS[0]));

class Ch70App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.04f, 0.04f, 0.08f};
        buildSimulatedScene();
    }

    void onShutdown() override {}

    void buildUi() override {
        interactive_.buildDebugPanel("第70章：MiniEngine 整合");
        ImGui::Separator();

        // 全屏大窗口
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
        ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus;
        if (!ImGui::Begin("##main", nullptr, flags)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("EngineTabs")) {

            // ── 代码量对比 ────────────────────────────────────────────────
            if (ImGui::BeginTabItem("代码量对比")) {
                ImGui::TextColored(ImVec4(1, 0.9f, 0.2f, 1), "ch60 裸 Vulkan（~1600 行）vs ch70 引擎封装（~300 行）");
                ImGui::Separator();

                // 总计
                int totalRaw = 0, totalEngine = 0;
                for (auto& c : CODE_COUNTS) {
                    totalRaw += c.rawLines;
                    totalEngine += c.engineLines;
                }
                float reduction = (1.0f - float(totalEngine) / float(totalRaw)) * 100.0f;
                ImGui::Text(
                    "总计：裸 Vulkan %d 行  →  引擎封装 %d 行  （减少 %.0f%%）", totalRaw, totalEngine, reduction);
                ImGui::Spacing();

                // 条形图
                for (int i = 0; i < N_MODULES; ++i) {
                    auto& c = CODE_COUNTS[i];
                    float rawFrac = float(c.rawLines) / 200.0f;
                    float engineFrac = float(c.engineLines) / 200.0f;
                    ImGui::Text("%-16s", c.name);
                    ImGui::SameLine();
                    // 裸 Vulkan（红色）
                    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.9f, 0.3f, 0.3f, 1));
                    ImGui::ProgressBar(rawFrac, ImVec2(200, 14), "");
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    ImGui::Text("%3d行", c.rawLines);
                    ImGui::SameLine(0, 12);
                    // 封装后（绿色）
                    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.3f, 0.9f, 0.4f, 1));
                    ImGui::ProgressBar(engineFrac, ImVec2(200, 14), "");
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    ImGui::Text("%3d行  ← %s", c.engineLines, c.module);
                }
                ImGui::EndTabItem();
            }

            // ── Application 框架 ──────────────────────────────────────────
            if (ImGui::BeginTabItem("Application 框架")) {
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "MiniEngine 使用方式");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "// 用户继承 Application，只关注业务逻辑：\n"
                    "class OutdoorDemo : public Application {\n"
                    "public:\n"
                    "    void onInit(MiniEngine& eng) override\n"
                    "    {\n"
                    "        // 注册着色器（1 行 / 程序）\n"
                    "        eng.shaders().registerProgram(\"scene\", \"scene.vert.spv\", \"scene.frag.spv\");\n\n"
                    "        // 加载纹理（1 行，自动缓存）\n"
                    "        eng.textures().load(\"assets/textures/ground.png\");\n\n"
                    "        // 注册材质（1 行）\n"
                    "        eng.materials().registerPBR(\"ground\", pipeline, layout);\n\n"
                    "        // 创建 ECS 实体（3 行）\n"
                    "        auto terrain = eng.world().createEntity(\"Terrain\");\n"
                    "        eng.world().add<TransformComponent>(terrain, {});\n"
                    "        eng.world().add<MeshComponent>(terrain, buildTerrainMesh());\n"
                    "    }\n\n"
                    "    void onUpdate(MiniEngine& eng, float dt) override\n"
                    "    {\n"
                    "        // 更新游戏逻辑（旋转、动画等）\n"
                    "        eng.world().forEach<TransformComponent>([&](auto id, auto& t){\n"
                    "            t.rotation.y += dt * 30.0f;\n"
                    "            t.dirty = true;\n"
                    "        });\n"
                    "    }\n\n"
                    "    void onRender(MiniEngine& eng, const FrameContext& ctx) override\n"
                    "    {\n"
                    "        // 遍历可见实体，提交 draw call\n"
                    "        eng.world().view<MeshComponent, MaterialComponent>(\n"
                    "            [&](auto id, auto& mesh, auto& mat) {\n"
                    "                eng.submit(id, pipelines[mat.materialId], descSets[id]);\n"
                    "            });\n"
                    "        // Engine 自动排序 + 提交\n"
                    "    }\n"
                    "};\n\n"
                    "int main() {\n"
                    "    MiniEngine engine;\n"
                    "    engine.run({\n"
                    "        .appName = \"我的游戏\",\n"
                    "        .width   = 1920, .height = 1080,\n"
                    "        .shaderDir = \"build/shaders\",\n"
                    "    }, OutdoorDemo{});\n"
                    "}\n");
                ImGui::EndTabItem();
            }

            // ── 子系统状态 ────────────────────────────────────────────────
            if (ImGui::BeginTabItem("子系统状态")) {
                ImGui::TextColored(ImVec4(0.8f, 0.5f, 1, 1), "各子系统当前状态");
                ImGui::Separator();

                auto row = [](const char* module, const char* status, const char* detail) {
                    ImGui::Text("%-25s", module);
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.3f, 1, 0.4f, 1), "%-12s", status);
                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", detail);
                };

                row("RHIDevice (ch61)", "✅ 就绪", "GPU 信息已查询");
                row("Buffer RAII (ch62)", "✅ 就绪", "StagingPool 已初始化");
                row("TextureCache (ch63)", "✅ 就绪", "0 张纹理已缓存");
                row("ShaderLibrary (ch64)", "✅ 就绪", "0 个程序已注册");
                row("PipelineCache (ch65)", "✅ 就绪", "pipeline.cache 已加载");
                row("DescriptorAllocator (ch66)", "✅ 就绪", "自动扩容 pool");
                row("CommandPool (ch67)", "✅ 就绪", "per-frame pool 已创建");
                row("World/ECS (ch68)", "✅ 就绪", (std::to_string(world_.entityCount()) + " 个实体").c_str());
                row("MaterialLibrary (ch69)", "✅ 就绪", "0 个材质模板");
                row("RenderGraph (ch51)", "✅ 就绪", "4 个瞬态资源");
                ImGui::Separator();

                ImGui::Text("模拟场景实体数 : %zu", world_.entityCount());
                if (ImGui::SliderInt("实体数量", &simEntityCount_, 10, 1000)) {
                    buildSimulatedScene();
                }
                ImGui::EndTabItem();
            }

            // ── 引擎架构 ──────────────────────────────────────────────────
            if (ImGui::BeginTabItem("引擎架构")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "MiniEngine 组件架构图");
                ImGui::Separator();
                ImGui::TextWrapped("          ┌────────────────────────────────────────────┐\n"
                                   "          │             Application（用户代码）           │\n"
                                   "          │  onInit / onUpdate / onRender / onResize    │\n"
                                   "          └────────────────────┬───────────────────────┘\n"
                                   "                               │\n"
                                   "          ┌────────────────────▼───────────────────────┐\n"
                                   "          │                MiniEngine                   │\n"
                                   "          │  ┌─────────┐ ┌──────────┐ ┌────────────┐  │\n"
                                   "          │  │RHIDevice│ │TextureCache│ │ShaderLibrary│  │\n"
                                   "          │  └─────────┘ └──────────┘ └────────────┘  │\n"
                                   "          │  ┌──────────────┐ ┌───────────────────┐   │\n"
                                   "          │  │MaterialLibrary│ │World (ECS)         │   │\n"
                                   "          │  └──────────────┘ └───────────────────┘   │\n"
                                   "          │  ┌───────────┐ ┌───────────┐ ┌────────┐  │\n"
                                   "          │  │PipelineCache│ │DrawCallBatch│ │RenderGraph│  │\n"
                                   "          │  └───────────┘ └───────────┘ └────────┘  │\n"
                                   "          └────────────────────────────────────────────┘\n"
                                   "                               │\n"
                                   "          ┌────────────────────▼───────────────────────┐\n"
                                   "          │              Vulkan API                     │\n"
                                   "          │  VkDevice / VkSwapchain / VkCommandBuffer   │\n"
                                   "          └────────────────────────────────────────────┘\n\n"
                                   "设计原则：\n"
                                   "  - 封装但不隐藏：每层都暴露底层句柄（getHandle()）\n"
                                   "  - 零运行时开销：封装层只在初始化/销毁时有开销\n"
                                   "  - 可选使用：不想用封装层时直接调用裸 Vulkan\n"
                                   "  - RAII 安全：所有 GPU 资源通过析构器自动释放\n");
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    engine::World world_;
    int simEntityCount_ = 100;

    void buildSimulatedScene() {
        world_ = engine::World{};
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> pos(-10.0f, 10.0f);
        for (int i = 0; i < simEntityCount_; ++i) {
            auto e = world_.createEntity();
            engine::TransformComponent tc{};
            tc.position = {pos(rng), pos(rng), pos(rng)};
            world_.add<engine::TransformComponent>(e, tc);
            world_.add<engine::MeshComponent>(e, {});
            world_.add<engine::MaterialComponent>(e, {static_cast<uint32_t>(i % 4)});
        }
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << " 第70章：MiniEngine 整合（引擎封装系列终章）\n";
    std::cout << " 引擎封装系列 — ch70/10\n";
    std::cout << "══════════════════════════════════════════════════════════\n\n";
    std::cout << "对比：ch60 裸 Vulkan ~1600 行  vs  引擎封装 ~300 行\n\n";
    try {
        Ch70App app;
        app.run("第70章：MiniEngine 整合", 1100, 780);
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
