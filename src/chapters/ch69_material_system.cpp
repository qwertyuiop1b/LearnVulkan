/**
 * @file ch69_material_system.cpp
 * @brief 第69章：材质系统（Material / MaterialInstance / MaterialLibrary）
 *
 * 【核心概念】
 *  Material     = ShaderProgram 引用 + 参数槽定义（模板，不持有数据）
 *  MaterialInstance = 继承 Material，覆盖具体参数值（持有 UBO + DescriptorSet）
 *  MaterialLibrary  = 注册/查找/实例化，统一管理生命周期
 *
 * 【价值】
 *  不同材质实例共享同一个 Pipeline，通过不同的 DescriptorSet 实现参数变化。
 *  只需切换描述符集，无需切换管线 → 高效渲染。
 *
 * 【API 设计】
 *  // 注册模板（一次）
 *  lib.registerPBR("pbr", pipeline, layout);
 *
 *  // 创建实例（多次，轻量）
 *  auto* stone = lib.instantiate("pbr");
 *  stone->set("albedo",   glm::vec4(0.5f, 0.5f, 0.5f, 1));
 *  stone->set("metallic", 0.0f);
 *  stone->set("roughness",0.8f);
 *
 *  auto* gold = lib.instantiate("pbr");
 *  gold->set("albedo",   glm::vec4(1.0f, 0.8f, 0.2f, 1));
 *  gold->set("metallic", 0.95f);
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <vulkan_tutorial/engine/material_system.hpp>

#include <algorithm>
#include <random>

using namespace engine;

class Ch69App : public DemoApp {
protected:
    void onInit() override
    {
        bgColor_ = {0.08f, 0.06f, 0.08f};
        setupMaterials();
    }

    void onShutdown() override {}


    void buildUi() override
    {
        interactive_.buildDebugPanel("第69章：材质系统");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 690), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Material System — Material / Instance / Library", nullptr))
        { ImGui::End(); return; }

        if (ImGui::BeginTabBar("MatTabs")) {

            if (ImGui::BeginTabItem("API 设计")) {
                ImGui::TextColored(ImVec4(1,0.85f,0.2f,1), "MaterialInstance — 参数化实例");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "// 注册 PBR 材质模板（含参数槽定义）\n"
                    "lib.registerPBR(\"pbr_opaque\", pipeline, pipelineLayout);\n\n"
                    "// 创建多个实例（共享同一 Pipeline）\n"
                    "auto* stone = lib.instantiate(\"pbr_opaque\");\n"
                    "stone->set(\"albedo\",    glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));\n"
                    "stone->set(\"metallic\",  0.0f);\n"
                    "stone->set(\"roughness\", 0.8f);\n\n"
                    "auto* gold = lib.instantiate(\"pbr_opaque\");\n"
                    "gold->set(\"albedo\",   glm::vec4(1.0f, 0.82f, 0.1f, 1.0f));\n"
                    "gold->set(\"metallic\", 0.98f);\n\n"
                    "// 每帧渲染前更新描述符（只对变化的 instance 更新）\n"
                    "stone->updateDescriptors(dev, allocator, cache, frameIndex);\n"
                    "gold->updateDescriptors(dev, allocator, cache, frameIndex);\n\n"
                    "// 渲染时绑定（切换 DescriptorSet，无需切换 Pipeline）\n"
                    "stone->bind(cmd, frameIndex);\n"
                    "stoneMesh.drawIndexed(cmd);\n"
                    "gold->bind(cmd, frameIndex);\n"
                    "goldMesh.drawIndexed(cmd);\n");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("实例参数调节")) {
                ImGui::TextColored(ImVec4(0.4f,1,0.5f,1), "实时调节 MaterialInstance 参数");
                ImGui::Separator();
                ImGui::Text("已注册材质模板数 : %d", (int)instanceDescs_.size() > 0 ? 1 : 0);
                ImGui::Text("已创建实例数量   : %d", (int)instanceDescs_.size());
                ImGui::Separator();

                for (int i = 0; i < (int)instanceDescs_.size(); ++i) {
                    auto& inst = instanceDescs_[i];
                    ImGui::PushID(i);
                    ImGui::Text("实例 [%d] %s", i, inst.label.c_str());
                    ImGui::ColorEdit4("颜色",    &inst.albedo.x);
                    ImGui::SliderFloat("金属度", &inst.metallic,  0.0f, 1.0f);
                    ImGui::SliderFloat("粗糙度", &inst.roughness, 0.0f, 1.0f);
                    ImGui::Separator();
                    ImGui::PopID();
                }
                ImGui::Spacing();
                if (ImGui::Button("添加实例")) {
                    std::mt19937 rng(static_cast<uint32_t>(instanceDescs_.size()*73));
                    InstanceDesc d;
                    d.label     = "实例 " + std::to_string(instanceDescs_.size());
                    d.albedo    = glm::vec4(float(rng()%100)/100.0f,
                                           float(rng()%100)/100.0f,
                                           float(rng()%100)/100.0f, 1.0f);
                    d.metallic  = float(rng()%100)/100.0f;
                    d.roughness = float(rng()%100)/100.0f;
                    instanceDescs_.push_back(d);
                }
                ImGui::SameLine();
                if (!instanceDescs_.empty() && ImGui::Button("删除最后一个")) {
                    instanceDescs_.pop_back();
                }
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("参数槽 & UBO")) {
                ImGui::TextColored(ImVec4(1,0.5f,0.8f,1), "PBRParams — 标准 PBR 参数布局（GPU std140）");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "struct PBRParams {\n"
                    "    alignas(16) vec4  albedo;         // rgba，a 用于透明度\n"
                    "    alignas( 4) float metallic;       // 0=绝缘体, 1=金属\n"
                    "    alignas( 4) float roughness;      // 0=镜面, 1=漫反射\n"
                    "    alignas( 4) float ao;             // 环境光遮蔽系数\n"
                    "    alignas( 4) float emissive;       // 发光强度\n"
                    "    alignas(16) vec4  emissiveColor;  // 发光颜色\n"
                    "};\n"
                    "// sizeof(PBRParams) = 48 bytes\n\n"
                    "// MaterialInstance 存储：\n"
                    "//   - params_: unordered_map<string, ParamValue>（覆盖值）\n"
                    "//   - ubos_[fi]: UniformBuffer（每帧一份，双缓冲）\n"
                    "//   - descSets_[fi]: DescriptorSet（含 UBO + 纹理）\n\n"
                    "// 每帧更新流程：\n"
                    "// 1. 把 params_ 中的 float/vec4 参数打包进 PBRParams\n"
                    "// 2. 写入当前帧的 UBO\n"
                    "// 3. 用 DescriptorBuilder 更新 DescriptorSet\n");
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

private:
    // (使用模拟数据，不需要实际 Vulkan 资源)

    struct InstanceDesc {
        std::string  label;
        glm::vec4    albedo   {1,1,1,1};
        float        metallic  = 0.0f;
        float        roughness = 0.5f;
    };
    std::vector<InstanceDesc> instanceDescs_;

    void setupMaterials()
    {
        instanceDescs_ = {
            {"石头",   {0.5f,0.5f,0.5f,1}, 0.0f, 0.8f},
            {"黄金",   {1.0f,0.8f,0.1f,1}, 0.98f,0.1f},
            {"塑料",   {0.2f,0.6f,1.0f,1}, 0.0f, 0.3f},
            {"铁锈",   {0.6f,0.3f,0.1f,1}, 0.7f, 0.9f},
        };
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第69章：材质系统（Material / MaterialInstance）\n";
    std::cout << " 引擎封装系列 — ch69/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try { Ch69App app; app.run("第69章：材质系统"); std::cout<<"✅ 正常退出\n"; }
    catch(const std::exception& e){ std::cerr<<"❌ "<<e.what()<<"\n"; return 1; }
}
