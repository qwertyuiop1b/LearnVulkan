/**
 * @file ch64_rhi_shader.cpp
 * @brief 第64章：着色器系统（Shader / ShaderProgram / ShaderLibrary）
 *
 * 【核心问题】
 *  每次创建 DescriptorSetLayout 和 PipelineLayout 都要手写：
 *    VkDescriptorSetLayoutBinding binding0{0, UNIFORM_BUFFER, 1, VERTEX, nullptr};
 *    VkDescriptorSetLayoutBinding binding1{1, COMBINED_SAMPLER, 1, FRAGMENT, nullptr};
 *    ...（重复几十遍）
 *
 * 【ShaderProgram 的解决方案】
 *  解析 SPIR-V 字节码中的 OpDecorate 指令，自动推导出：
 *    - 每个 binding 的 set / binding / type / stage
 *    - 自动合并多个阶段的 binding 信息
 *    - 生成 VkDescriptorSetLayout 和 VkPipelineLayout
 *
 * 【ShaderLibrary 热重载】
 *  lib.setReloadCallback([&](const string& name, ShaderProgram*) {
 *      rebuildPipeline(name);   // 着色器变化时重建管线
 *  });
 *  lib.reloadAll();   // 扫描文件时间戳，有变化则重载
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <vulkan_tutorial/engine/rhi_shader.hpp>

#include <chrono>
#include <filesystem>
#include <sstream>

class Ch64App : public DemoApp {
protected:
    void onInit() override
    {
        bgColor_ = {0.06f, 0.04f, 0.10f};
        simulateShaderLibrary();
    }

    void buildUi() override
    {
        interactive_.buildDebugPanel("第64章：着色器系统");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 690), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Shader 封装 — Shader / ShaderProgram / ShaderLibrary", nullptr))
        { ImGui::End(); return; }

        if (ImGui::BeginTabBar("ShaderTabs")) {

            if (ImGui::BeginTabItem("ShaderProgram API")) {
                ImGui::TextColored(ImVec4(1,0.8f,0.2f,1), "ShaderProgram — 自动推导 DescriptorSetLayout");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "// 传统写法：手工写每个 binding 描述（约 40 行 / 程序）\n"
                    "VkDescriptorSetLayoutBinding b0{};\n"
                    "b0.binding = 0;\n"
                    "b0.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;\n"
                    "b0.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;\n"
                    "// ... 每个 binding 重复 4 行 ...\n\n"
                    "// ShaderProgram 封装：\n"
                    "ShaderProgram prog;\n"
                    "prog.addStage(dev, \"pbr.vert.spv\", VK_SHADER_STAGE_VERTEX_BIT);\n"
                    "prog.addStage(dev, \"pbr.frag.spv\", VK_SHADER_STAGE_FRAGMENT_BIT);\n"
                    "prog.link(dev);\n"
                    "// 自动生成了：\n"
                    "prog.pipelineLayout()           // 直接用于管线创建\n"
                    "prog.descriptorSetLayout(0)     // 直接用于 VkDescriptorSetAllocateInfo\n"
                    "prog.allBindings()              // 所有 binding 信息（调试用）\n\n"
                    "// 配合 PipelineBuilder：\n"
                    "GraphicsPipelineBuilder(dev)\n"
                    "    .setProgram(prog)           // 一行设置所有 stage + layout\n"
                    "    ...\n"
                    "    .build(pipelineCache);\n");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("SPIR-V 反射")) {
                ImGui::TextColored(ImVec4(0.5f,1,0.5f,1), "SPIR-V 反射原理（简化版）");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "SPIR-V 是一种二进制中间语言格式。\n"
                    "每个 SPIR-V 指令格式：[wordCount:16|opcode:16] [word1] [word2] ...\n\n"
                    "关键指令（Shader 反射需要解析）：\n\n"
                    "  OpDecorate id 33 set      → 资源属于第几个 descriptor set\n"
                    "  OpDecorate id 34 binding  → 资源在 set 中的 binding 编号\n"
                    "  OpVariable id type storageClass → 资源类型\n"
                    "    storageClass = 0: UniformConstant（sampler）\n"
                    "    storageClass = 2: Uniform（UBO）\n"
                    "    storageClass = 12: StorageBuffer（SSBO）\n\n"
                    "解析流程：\n"
                    "  1. 跳过 5 个 header word\n"
                    "  2. 逐指令扫描\n"
                    "  3. 遇到 OpDecorate → 记录 id→set 或 id→binding\n"
                    "  4. 遇到 OpVariable → 记录 id→storageClass\n"
                    "  5. 合并 id 信息 → BindingReflect 数组\n\n"
                    "注意：工业级反射使用 SPIRV-Cross 或 spirv-reflect 库，\n"
                    "可以获取更多信息（变量名、结构体成员、push constant 布局等）。");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("ShaderLibrary 热重载")) {
                ImGui::TextColored(ImVec4(0.8f,0.5f,1,1), "ShaderLibrary — 着色器热重载");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "ShaderLibrary lib;\n"
                    "lib.init(dev, \"build/shaders\");\n\n"
                    "// 注册着色器程序：\n"
                    "lib.registerProgram(\"pbr\",    \"pbr.vert.spv\",    \"pbr.frag.spv\");\n"
                    "lib.registerProgram(\"skybox\", \"skybox.vert.spv\", \"skybox.frag.spv\");\n"
                    "lib.registerCompute(\"ssao\",   \"ssao.comp.spv\");\n\n"
                    "// 使用：\n"
                    "ShaderProgram* prog = lib.get(\"pbr\");\n\n"
                    "// 热重载（检测文件修改时间戳）：\n"
                    "lib.setReloadCallback([&](const string& name, ShaderProgram* prog) {\n"
                    "    // 着色器重载后，需要重建使用该程序的管线\n"
                    "    rebuildPipelinesFor(name);\n"
                    "});\n\n"
                    "// 在主循环中（如按 F5）：\n"
                    "bool reloaded = lib.reloadAll();\n"
                    "if (reloaded) { log(\"着色器已重载！\"); }\n\n"
                    "// 单个重载：\n"
                    "lib.reloadIfDirty(\"pbr\");  // 只检查 pbr 程序的文件\n");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.5f,1,0.8f,1), "模拟已注册程序列表");
                ImGui::Separator();
                for (const auto& p : programs_) {
                    ImGui::Text("  %-20s  状态：%s  绑定数：%d",
                        p.name.c_str(), p.dirty ? "⚠️ 需重载" : "✅ 最新", p.bindingCount);
                }
                ImGui::Spacing();
                if (ImGui::Button("按 F5 重载所有")) {
                    for (auto& p : programs_) p.dirty = false;
                    ++reloadCount_;
                }
                ImGui::SameLine();
                if (ImGui::Button("模拟文件修改（标记为 dirty）")) {
                    if (!programs_.empty()) programs_[0].dirty = true;
                }
                ImGui::Text("总重载次数：%d", reloadCount_);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

private:
    struct ProgramEntry { std::string name; int bindingCount; bool dirty; };
    std::vector<ProgramEntry> programs_;
    int reloadCount_ = 0;

    void simulateShaderLibrary()
    {
        programs_ = {
            {"pbr",        5, false},
            {"skybox",     2, false},
            {"ssao_blur",  2, false},
            {"shadow",     1, false},
            {"composite",  3, false},
        };
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第64章：着色器系统（ShaderProgram / ShaderLibrary）\n";
    std::cout << " 引擎封装系列 — ch64/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try { Ch64App app; app.run("第64章：着色器系统"); std::cout<<"✅ 正常退出\n"; }
    catch(const std::exception& e){ std::cerr<<"❌ "<<e.what()<<"\n"; return 1; }
}
