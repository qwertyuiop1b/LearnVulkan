/**
 * @file ch65_pipeline_builder.cpp
 * @brief 第65章：管线构建器（GraphicsPipelineBuilder / PipelineCache）
 *
 * 【VkGraphicsPipelineCreateInfo 的痛点】
 *  创建一条图形管线需要填写 30+ 个字段，每次都要 80 行代码。
 *  当需要 3 种变体（实心/线框/无剔除）时，传统写法需要 240 行重复代码。
 *
 * 【GraphicsPipelineBuilder 流式 API】
 *  GraphicsPipelineBuilder(dev)
 *      .setProgram(shaderProgram)
 *      .setVertexInput<MyVertex>()
 *      .setDepthTest(true)
 *      .setCullMode(VK_CULL_MODE_BACK_BIT)
 *      .setRenderPass(renderPass)
 *      .build(cache.vkCache());   // ← 1 行
 *
 * 【PipelineCache 两层缓存】
 *  1. VkPipelineCache（Vulkan 原生）— 跨 session 缓存 SPIR-V 编译结果，
 *     序列化到 pipeline.cache 文件，下次启动秒加载，无需重新编译
 *  2. 哈希 → VkPipeline（运行时）— 避免相同状态重复创建管线对象
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <vulkan_tutorial/engine/pipeline_builder.hpp>

#include <chrono>

class Ch65App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.08f, 0.05f, 0.05f};
        // 模拟管线创建统计
        simulatePipelineCreation();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第65章：管线构建器");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 690), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("PipelineBuilder — 流式 API + 两层缓存", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("PipeTabs")) {

            if (ImGui::BeginTabItem("GraphicsPipelineBuilder")) {
                ImGui::TextColored(ImVec4(1, 0.8f, 0.2f, 1), "GraphicsPipelineBuilder 流式 API");
                ImGui::Separator();
                ImGui::TextWrapped("// 传统写法（约 80 行，每种管线都要重复）：\n"
                                   "VkPipelineVertexInputStateCreateInfo vi{};\n"
                                   "vi.sType = ...;\n"
                                   "vi.vertexBindingDescriptionCount = ...;\n"
                                   "// ... 5 个 CreateInfo struct，每个 5-10 行 ...\n\n"
                                   "// GraphicsPipelineBuilder（5 行）：\n"
                                   "VkPipeline solid = GraphicsPipelineBuilder(dev)\n"
                                   "    .setProgram(shaderProg)\n"
                                   "    .setVertexInput<SceneVertex>()\n"
                                   "    .setDepthTest(true, true)\n"
                                   "    .setCullMode(VK_CULL_MODE_BACK_BIT)\n"
                                   "    .setRenderPass(renderPass, 0)\n"
                                   "    .build(cache.vkCache());\n\n"
                                   "// 线框变体（1 行修改）：\n"
                                   "VkPipeline wire = GraphicsPipelineBuilder(dev)\n"
                                   "    .setProgram(shaderProg)\n"
                                   "    .setVertexInput<SceneVertex>()\n"
                                   "    .setPolygonMode(VK_POLYGON_MODE_LINE)\n"
                                   "    .setDepthTest(false)\n"
                                   "    .setRenderPass(renderPass)\n"
                                   "    .build(cache.vkCache());\n\n"
                                   "// 透明混合（额外 1 行）：\n"
                                   "VkPipeline alpha = GraphicsPipelineBuilder(dev)\n"
                                   "    .setProgram(shaderProg)\n"
                                   "    .setVertexInput<SceneVertex>()\n"
                                   "    .setAlphaBlend(true)\n"
                                   "    .setDepthTest(true, false)   // 测试深度，不写深度\n"
                                   "    .setRenderPass(renderPass)\n"
                                   "    .build(cache.vkCache());\n");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("PipelineCache")) {
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "PipelineCache 两层设计");
                ImGui::Separator();
                ImGui::TextWrapped("// 第一层：VkPipelineCache（Vulkan 原生，跨 session）\n"
                                   "PipelineCache cache;\n"
                                   "cache.init(dev, \"pipeline.cache\");\n"
                                   "// 第一次启动：从头编译 SPIR-V，耗时 ~200ms\n"
                                   "// 之后启动：从 pipeline.cache 文件加载，耗时 ~5ms\n\n"
                                   "// 程序退出前保存：\n"
                                   "cache.save();   // 序列化到 pipeline.cache 文件\n\n"
                                   "// 第二层：哈希 → VkPipeline（运行时去重）\n"
                                   "size_t hash = computePipelineHash(program, state);\n"
                                   "VkPipeline cached = cache.find(hash);\n"
                                   "if (cached == VK_NULL_HANDLE) {\n"
                                   "    cached = builder.build(cache.vkCache());\n"
                                   "    cache.store(hash, cached);\n"
                                   "}\n");
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "管线创建统计（当前会话）");
                ImGui::Separator();
                ImGui::Text("总管线数量   : %d", stats_.totalPipelines);
                ImGui::Text("缓存命中次数 : %d", stats_.cacheHits);
                ImGui::Text("冷创建次数   : %d", stats_.coldCreations);
                ImGui::Text("创建耗时节省 : %.1f ms（估算）", float(stats_.cacheHits) * 180.0f);
                ImGui::Spacing();
                ImGui::Text("pipeline.cache 文件大小 : %.1f KB", float(stats_.cacheFileSizeBytes) / 1024.0f);
                ImGui::Text("下次启动加速 : ✅ 已缓存 %d 条管线", stats_.totalPipelines);
                ImGui::Spacing();
                if (ImGui::Button("模拟新管线创建（冷创建）")) {
                    ++stats_.totalPipelines;
                    ++stats_.coldCreations;
                    stats_.cacheFileSizeBytes += 4096;
                }
                ImGui::SameLine();
                if (ImGui::Button("模拟管线缓存命中")) {
                    ++stats_.cacheHits;
                }
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("VertexInputState")) {
                ImGui::TextColored(ImVec4(1, 0.5f, 0.8f, 1), "setVertexInput<T>() — 类型安全的顶点输入");
                ImGui::Separator();
                ImGui::TextWrapped("// 顶点结构体中声明静态方法：\n"
                                   "struct SceneVertex {\n"
                                   "    glm::vec3 pos;\n"
                                   "    glm::vec3 normal;\n"
                                   "    glm::vec2 uv;\n\n"
                                   "    static VertexInputState vertexInputState() {\n"
                                   "        VertexInputState vis;\n"
                                   "        vis.bindings = { {0, sizeof(SceneVertex), PER_VERTEX} };\n"
                                   "        vis.attributes = {\n"
                                   "            {0, 0, R32G32B32_SFLOAT, offsetof(SceneVertex, pos)},\n"
                                   "            {1, 0, R32G32B32_SFLOAT, offsetof(SceneVertex, normal)},\n"
                                   "            {2, 0, R32G32_SFLOAT,    offsetof(SceneVertex, uv)},\n"
                                   "        };\n"
                                   "        return vis;\n"
                                   "    }\n"
                                   "};\n\n"
                                   "// Builder 中使用：\n"
                                   "builder.setVertexInput<SceneVertex>();   // 自动从类型推导\n\n"
                                   "// 对比传统写法：需要手写 binding + attribute 数组\n"
                                   "// 每次改变顶点格式，忘记更新会导致 crash\n"
                                   "// 使用 offsetof() 和模板，编译时保证类型安全\n");
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    struct PipeStats {
        int totalPipelines = 12;
        int cacheHits = 8;
        int coldCreations = 4;
        int cacheFileSizeBytes = 49152;
    } stats_;

    void simulatePipelineCreation() {
        stats_ = {12, 8, 4, 49152};
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第65章：管线构建器（PipelineBuilder / PipelineCache）\n";
    std::cout << " 引擎封装系列 — ch65/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch65App app;
        app.run("第65章：管线构建器");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
