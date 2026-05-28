/**
 * @file ch88_dynamic_weather.cpp
 * @brief 第88章：动态天气系统（Rain / Snow / Wind）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【架构】
 *
 *  WeatherStateMachine — 晴/阴/雨/雪/暴风雨 状态切换
 *  ParticleWeatherFX   — GPU 粒子（复用 ch52 发射器）
 *  WetnessSystem       — 全局湿润度 → 修改 PBR roughness
 *  WindField           — 3D 噪声驱动植被/粒子/水面
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <imgui.h>
#include <glm/glm.hpp>
#include <cmath>
#include <array>
#include <iostream>

enum class WeatherType { Clear, Cloudy, Rain, Snow, Storm };

class Ch88App : public DemoApp {
protected:
    void onInit() override
    {
        bgColor_ = {0.08f, 0.10f, 0.14f};
        applyWeather(currentWeather_);
    }

    void onUpdate() override
    {
        elapsed_ += 0.016f;
        if (autoCycle_) {
            cycleTimer_ += 0.016f;
            if (cycleTimer_ > 8.0f) {
                cycleTimer_ = 0.0f;
                int next = (static_cast<int>(currentWeather_) + 1) % 5;
                setWeather(static_cast<WeatherType>(next));
            }
        }
        updateParticles();
    }

    void buildUi() override
    {
        interactive_.buildDebugPanel("第88章：动态天气");
        ImGui::Separator();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第88章：动态天气系统", nullptr))
        { ImGui::End(); return; }

        if (ImGui::BeginTabBar("WeatherTabs")) {
            if (ImGui::BeginTabItem("天气状态")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "WeatherStateMachine");
                ImGui::Separator();
                const char* names[] = {"晴天", "阴天", "雨天", "雪天", "暴风雨"};
                int w = static_cast<int>(currentWeather_);
                if (ImGui::Combo("当前天气", &w, names, 5))
                    setWeather(static_cast<WeatherType>(w));
                ImGui::Checkbox("自动循环切换", &autoCycle_);
                ImGui::SliderFloat("过渡时间 (s)", &transitionTime_, 0.5f, 10.0f, "%.1f");
                ImGui::Spacing();
                ImGui::Text("过渡进度 : %.0f%%", transitionProgress_ * 100.0f);
                ImGui::ProgressBar(transitionProgress_, ImVec2(-1, 18));
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("粒子与风")) {
                ImGui::SliderFloat3("风向", &windDir_.x, -1.0f, 1.0f);
                ImGui::SliderFloat("风速 (m/s)", &windSpeed_, 0.0f, 30.0f, "%.1f");
                ImGui::SliderInt("活跃粒子数", &activeParticles_, 0, maxParticles_);
                ImGui::SliderFloat("粒子密度", &particleDensity_, 0.0f, 1.0f, "%.2f");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.4f, 1, 0.5f, 1), "GPU 粒子参数（rain.comp）：");
                ImGui::Text("  maxParticles  = %d", maxParticles_);
                ImGui::Text("  spawnRate     = %.0f /s", spawnRate_);
                ImGui::Text("  gravity       = (%.1f, %.1f, %.1f)",
                    gravity_.x, gravity_.y, gravity_.z);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("湿润度系统")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "Wetness → PBR Roughness");
                ImGui::Separator();
                ImGui::TextWrapped(
                    "全局湿润度影响所有材质：\n"
                    "  wetRoughness = mix(dryRoughness, 0.05, wetness)\n"
                    "  specBoost    = wetness × 0.3\n\n"
                    "积水效果（雨天）：\n"
                    "  puddleMask = smoothstep(0.0, 0.1, worldNormal.y)\n"
                    "             × wetness × noise(worldXZ)");
                ImGui::SliderFloat("湿润度", &wetness_, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("干燥粗糙度", &dryRoughness_, 0.3f, 1.0f, "%.2f");
                float wetRough = dryRoughness_ * (1.0f - wetness_) + 0.05f * wetness_;
                ImGui::Text("当前有效粗糙度 : %.3f", wetRough);
                ImGui::ProgressBar(wetness_, ImVec2(-1, 16), "湿润覆盖");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("集成架构")) {
                ImGui::TextWrapped(
                    "class WeatherSystem {\n"
                    "    WeatherStateMachine state_;\n"
                    "    ParticleEmitter     rainEmitter_;\n"
                    "    ParticleEmitter     snowEmitter_;\n"
                    "    float               wetness_ = 0;\n"
                    "    glm::vec3           windDir_;\n\n"
                    "    void update(float dt) {\n"
                    "        state_.update(dt);\n"
                    "        wetness_ = lerp(wetness_, state_.targetWetness(), dt * 0.1f);\n"
                    "        rainEmitter_.setActive(state_.isRaining());\n"
                    "        applyGlobalUniform(wetness_, windDir_);\n"
                    "    }\n"
                    "};");
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

private:
    WeatherType currentWeather_ = WeatherType::Clear;
    float transitionTime_ = 3.0f;
    float transitionProgress_ = 1.0f;
    float cycleTimer_ = 0.0f;
    bool autoCycle_ = false;
    float elapsed_ = 0.0f;
    glm::vec3 windDir_{1.0f, 0.0f, 0.3f};
    float windSpeed_ = 5.0f;
    int activeParticles_ = 0;
    int maxParticles_ = 10000;
    float particleDensity_ = 0.0f;
    float spawnRate_ = 0.0f;
    glm::vec3 gravity_{0.0f, -9.8f, 0.0f};
    float wetness_ = 0.0f;
    float dryRoughness_ = 0.7f;

    void setWeather(WeatherType w)
    {
        currentWeather_ = w;
        transitionProgress_ = 0.0f;
        applyWeather(w);
    }

    void applyWeather(WeatherType w)
    {
        switch (w) {
        case WeatherType::Clear:
            bgColor_ = {0.15f, 0.20f, 0.35f};
            maxParticles_ = 0; spawnRate_ = 0; wetness_ = 0.0f;
            gravity_ = {0, -9.8f, 0}; particleDensity_ = 0;
            break;
        case WeatherType::Cloudy:
            bgColor_ = {0.10f, 0.12f, 0.18f};
            maxParticles_ = 0; spawnRate_ = 0; wetness_ = 0.1f;
            particleDensity_ = 0;
            break;
        case WeatherType::Rain:
            bgColor_ = {0.06f, 0.08f, 0.12f};
            maxParticles_ = 8000; spawnRate_ = 2000; wetness_ = 0.8f;
            gravity_ = {0, -15.0f, 0}; particleDensity_ = 0.7f;
            break;
        case WeatherType::Snow:
            bgColor_ = {0.12f, 0.14f, 0.18f};
            maxParticles_ = 5000; spawnRate_ = 800; wetness_ = 0.3f;
            gravity_ = {0, -2.0f, 0}; particleDensity_ = 0.5f;
            break;
        case WeatherType::Storm:
            bgColor_ = {0.03f, 0.04f, 0.08f};
            maxParticles_ = 15000; spawnRate_ = 5000; wetness_ = 1.0f;
            gravity_ = {0, -20.0f, 0}; particleDensity_ = 1.0f;
            windSpeed_ = 25.0f;
            break;
        }
        activeParticles_ = maxParticles_;
    }

    void updateParticles()
    {
        if (transitionProgress_ < 1.0f)
            transitionProgress_ = std::min(1.0f, transitionProgress_ + 0.016f / transitionTime_);
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第88章：动态天气系统\n";
    std::cout << " 游戏 Demo 系列 — ch88/6\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch88App app;
        app.run("第88章：动态天气系统");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
