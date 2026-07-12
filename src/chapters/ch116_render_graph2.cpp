/** @file ch116_render_graph2.cpp @brief DAG compilation, culling and resource alias planning. */

#include <vulkan_tutorial/engine/engineering_gpu_chapter.hpp>
#include <vulkan_tutorial/render_graph2.hpp>

#include <imgui.h>
#include <iostream>

class Ch116App final : public EngineeringGpuChapterApp {
  protected:
    const char* chapterTitle() const override { return "第116章：Render Graph 2.0"; }
    uint32_t engineeringMode() const override { return 116; }

    void onEngineeringInit() override {
        const uint32_t shadow = graph_.addResource("shadow");
        const uint32_t depth = graph_.addResource("depth");
        const uint32_t hdr = graph_.addResource("hdr");
        const uint32_t bloom = graph_.addResource("bloom");
        const uint32_t unused = graph_.addResource("unused_debug");
        graph_.addPass({"shadow", {}, {shadow}, false, {}});
        graph_.addPass({"depth", {}, {depth}, false, {}});
        graph_.addPass({"scene", {shadow, depth}, {hdr}, true, {}});
        graph_.addPass({"bloom", {hdr}, {bloom}, false, {}});
        graph_.addPass({"unused", {}, {unused}, false, {}});
        graph_.markOutput(bloom);
        compiled_ = graph_.compile();
    }

    void buildChapterUi() override {
        ImGui::SetNextWindowPos({12, 315}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Render Graph 2.0", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("DAG compile: %s", compiled_ ? "valid" : "cycle detected");
            ImGui::Text("Passes: %zu  live: %zu  aliases: %zu", graph_.order().size(), graph_.liveOrder().size(),
                        graph_.aliasCount());
            for (uint32_t pass : graph_.liveOrder())
                ImGui::BulletText("%s", graph_.pass(pass).name.c_str());
            ImGui::TextUnformatted("unused_debug is culled; transient lifetimes share alias slots.");
        }
        ImGui::End();
    }

    glm::vec4 engineeringParameters() const override {
        return {compiled_ ? 1.0f : 0.0f, static_cast<float>(graph_.liveOrder().size()),
                static_cast<float>(graph_.aliasCount()), 0.0f};
    }

  private:
    vulkan_tutorial::RenderGraph2 graph_;
    bool compiled_ = false;
};

int main() {
    try { Ch116App app; app.run("ch116 - Render Graph 2.0", 1280, 720); }
    catch (const std::exception& error) { std::cerr << "ch116 failed: " << error.what() << '\n'; return 1; }
    return 0;
}
