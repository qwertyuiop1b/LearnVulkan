/**
 * @file ch83_restir.cpp
 * @brief 第83章：ReSTIR（Reservoir Spatio-Temporal Importance Resampling）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【ReSTIR — 2020 SIGGRAPH 最重要论文之一】
 *
 *  论文：Spatiotemporal reservoir resampling for real-time ray tracing with
 *        dynamic direct lighting（Bitterli et al., 2020）
 *
 *  核心问题：路径追踪每像素只能追踪 1-2 条光线（实时预算），
 *            如何让稀疏样本看起来像 64spp？
 *
 *  ReSTIR 的洞察：
 *    - 用 Weighted Reservoir Sampling（WRS）高效维护"最佳样本"
 *    - 在时间维度（上一帧）和空间维度（邻居像素）重用样本
 *    - 理论等效：每像素采样 1000+ 光源，代价只有 1spp
 *
 *  ReSTIR GI / ReSTIR PT：
 *    - 原版 ReSTIR 只处理直接光照（Direct Illumination）
 *    - ReSTIR GI（2021）：扩展到间接光照（Global Illumination）
 *    - ReSTIR PT（2022）：扩展到完整路径追踪（Path Tracing）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <vulkan_tutorial/engine/demo_app.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

// ─── Reservoir 数据结构（CPU 端模拟） ─────────────────────────────────────────

struct LightSample {
    float position[3]; ///< 光源位置
    float intensity;   ///< 光源强度
    int lightId;       ///< 光源索引
};

/// @brief Weighted Reservoir Sampling 的核心数据结构
struct Reservoir {
    LightSample y{};  ///< 当前最优样本
    float wSum = 0.f; ///< 累计权重之和
    int M = 0;        ///< 已处理的候选样本数
    float W = 0.f;    ///< 无偏权重（shading 时使用）

    void reset() {
        wSum = 0.f;
        M = 0;
        W = 0.f;
    }
};

// ─── 采样算法模拟器 ───────────────────────────────────────────────────────────

class SamplingSimulator {
  public:
    explicit SamplingSimulator(int lightCount, int pixelCount)
        : lightCount_(lightCount), pixelCount_(pixelCount), rng_(42), reservoirs_(size_t(pixelCount)),
          prevReservoirs_(size_t(pixelCount)), uniformErrors_(MAX_FRAMES), restirErrors_(MAX_FRAMES) {}

    /// @brief 执行一帧的 ReSTIR 更新，返回当前帧估算误差
    float stepRestir() {
        generateLights();
        float totalErr = 0.f;
        for (int p = 0; p < pixelCount_; ++p) {
            totalErr += processPixelRestir(p);
        }
        prevReservoirs_ = reservoirs_;
        float err = totalErr / float(pixelCount_);
        if (frameIndex_ < MAX_FRAMES)
            restirErrors_[frameIndex_] = err;
        return err;
    }

    /// @brief 执行一帧的均匀采样，返回当前帧估算误差
    float stepUniform() {
        generateLights();
        float totalErr = 0.f;
        for (int p = 0; p < pixelCount_; ++p) {
            totalErr += processPixelUniform(p);
        }
        float err = totalErr / float(pixelCount_);
        if (frameIndex_ < MAX_FRAMES)
            uniformErrors_[frameIndex_] = err;
        ++frameIndex_;
        return err;
    }

    void reset() {
        frameIndex_ = 0;
        std::fill(uniformErrors_.begin(), uniformErrors_.end(), 0.f);
        std::fill(restirErrors_.begin(), restirErrors_.end(), 0.f);
        for (auto& r : reservoirs_)
            r.reset();
        for (auto& r : prevReservoirs_)
            r.reset();
    }

    const std::vector<float>& uniformErrors() const {
        return uniformErrors_;
    }
    const std::vector<float>& restirErrors() const {
        return restirErrors_;
    }
    int frameCount() const {
        return std::min(frameIndex_, MAX_FRAMES);
    }

    static constexpr int MAX_FRAMES = 200;

  private:
    int lightCount_;
    int pixelCount_;
    std::mt19937 rng_;
    std::vector<Reservoir> reservoirs_;
    std::vector<Reservoir> prevReservoirs_;
    std::vector<float> uniformErrors_;
    std::vector<float> restirErrors_;
    std::vector<LightSample> lights_;
    int frameIndex_ = 0;

    void generateLights() {
        std::uniform_real_distribution<float> pos(-10.f, 10.f);
        std::uniform_real_distribution<float> intensity(0.1f, 5.f);
        lights_.resize(size_t(lightCount_));
        for (auto& l : lights_) {
            l.position[0] = pos(rng_);
            l.position[1] = pos(rng_) + 5.f;
            l.position[2] = pos(rng_);
            l.intensity = intensity(rng_);
        }
    }

    /// @brief Weighted Reservoir Sampling 更新函数
    void updateReservoir(Reservoir& r, const LightSample& x, float w) {
        r.wSum += w;
        ++r.M;
        std::uniform_real_distribution<float> u(0.f, 1.f);
        if (u(rng_) < w / r.wSum)
            r.y = x;
    }

    /// @brief 合并两个 Reservoir（跨像素/跨帧重用）
    void combineReservoirs(Reservoir& dst, const Reservoir& src) {
        if (src.M == 0)
            return;
        updateReservoir(dst, src.y, src.W * float(src.M));
        dst.M += src.M;
    }

    float computeTargetPdf(const LightSample& s) {
        return s.intensity / (1.f + s.position[0] * s.position[0] + s.position[1] * s.position[1]);
    }

    float processPixelRestir(int pixelId) {
        Reservoir& r = reservoirs_[size_t(pixelId)];
        r.reset();

        // ① 初始候选采样（1spp）
        std::uniform_int_distribution<int> lightDist(0, lightCount_ - 1);
        int idx = lightDist(rng_);
        float w = computeTargetPdf(lights_[size_t(idx)]) * float(lightCount_);
        updateReservoir(r, lights_[size_t(idx)], w);

        // ② 时间复用：与上一帧同像素合并
        combineReservoirs(r, prevReservoirs_[size_t(pixelId)]);

        // ③ 空间复用：与左右邻居合并
        if (pixelId > 0)
            combineReservoirs(r, reservoirs_[size_t(pixelId - 1)]);
        if (pixelId < pixelCount_ - 1)
            combineReservoirs(r, reservoirs_[size_t(pixelId + 1)]);

        // ④ 计算无偏权重
        float pdf = computeTargetPdf(r.y);
        r.W = (pdf > 0.f) ? r.wSum / (pdf * float(r.M)) : 0.f;

        // ⑤ 误差 = 和理想值的差（理想值 = 所有光源的总贡献归一化）
        float idealContrib = 0.f;
        for (const auto& l : lights_)
            idealContrib += computeTargetPdf(l);
        idealContrib /= float(lightCount_);
        float estimate = pdf * r.W;
        return std::fabs(estimate - idealContrib);
    }

    float processPixelUniform(int pixelId) {
        (void)pixelId;
        // 均匀采样：随机选 1 个光源
        std::uniform_int_distribution<int> d(0, lightCount_ - 1);
        int idx = d(rng_);
        float pdf = computeTargetPdf(lights_[size_t(idx)]);
        float est = pdf * float(lightCount_);

        float idealContrib = 0.f;
        for (const auto& l : lights_)
            idealContrib += computeTargetPdf(l);
        idealContrib /= float(lightCount_);
        return std::fabs(est - idealContrib);
    }
};

// ─── 应用类 ───────────────────────────────────────────────────────────────────

class Ch83App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.05f, 0.04f, 0.10f};
        simulator_ = std::make_unique<SamplingSimulator>(lightCount_, pixelCount_);
    }

    void onUpdate() override {
        if (isRunning_) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - lastTime_).count();
            lastTime_ = now;
            accTime_ += dt;
            if (accTime_ >= stepInterval_) {
                accTime_ = 0.f;
                latestUniformErr_ = simulator_->stepUniform();
                latestRestirErr_ = simulator_->stepRestir();
            }
        }
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第83章：ReSTIR");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 680), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("第83章：ReSTIR（Reservoir Spatio-Temporal Importance Resampling）", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("RestirTabs")) {
            buildTabBackground();
            buildTabReservoir();
            buildTabAlgorithm();
            buildTabSimulation();
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    std::unique_ptr<SamplingSimulator> simulator_;

    int lightCount_ = 64;
    int pixelCount_ = 128;
    bool isRunning_ = false;
    float stepInterval_ = 0.05f;
    float accTime_ = 0.f;
    float latestUniformErr_ = 0.f;
    float latestRestirErr_ = 0.f;
    std::chrono::steady_clock::time_point lastTime_ = std::chrono::steady_clock::now();

    void buildTabBackground() {
        if (!ImGui::BeginTabItem("问题背景"))
            return;

        ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "路径追踪的实时困境与 ReSTIR 的解法");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "核心矛盾：");
        ImGui::BulletText("实时路径追踪：每像素只能发射 1-2 条光线（16ms 预算）");
        ImGui::BulletText("1 spp（样本/像素）的图像噪点极多，无法直接展示");
        ImGui::BulletText("离线渲染：通常需要 64-4096 spp 才能收敛");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "传统优化方向的局限：");
        ImGui::TextWrapped("// 方法 1：直接从光源采样（Direct Light Sampling）\n"
                           "// 场景有 1000 个面光源时，随机选 1 个，大多数贡献接近 0\n"
                           "// 方差极高，噪点严重\n\n"
                           "// 方法 2：重要性采样（Importance Sampling）\n"
                           "// 根据 BRDF / 光源亮度加权选择采样方向\n"
                           "// 减少方差，但单帧样本数仍然很少\n\n"
                           "// 方法 3：时间累积（TAA / Temporal Accumulation）\n"
                           "// 利用上一帧结果降噪，但快速运动时出现鬼影\n");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "ReSTIR 的核心洞察：");
        ImGui::BulletText("\"好的样本\" 在时间和空间上有局部相关性");
        ImGui::BulletText("相邻像素的最优采样光源往往相似（空间重用）");
        ImGui::BulletText("前一帧的最优样本在当前帧大概率仍然有效（时间重用）");
        ImGui::BulletText("用 Reservoir 数据结构，O(1) 合并任意多个样本流");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1), "效果：");
        ImGui::BulletText("时间重用：等效积累 30+ 帧样本（指数级增长）");
        ImGui::BulletText("空间重用：每像素等效采样 16-64 个光源");
        ImGui::BulletText("合计：1 spp 代价，≈1000 spp 效果");
        ImGui::BulletText("NVIDIA RTX 显卡已将 ReSTIR 内置到驱动层");

        ImGui::EndTabItem();
    }

    void buildTabReservoir() {
        if (!ImGui::BeginTabItem("Reservoir 数据结构"))
            return;

        ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1), "Weighted Reservoir Sampling（WRS）");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextWrapped("// Reservoir = 一个像素当前维护的\"最优光源候选\"\n"
                           "struct Reservoir {\n"
                           "    LightSample y;      // 当前最优样本（最有可能对此像素有贡献的光源）\n"
                           "    float       wSum;   // 已处理所有候选样本的权重之和\n"
                           "    int         M;      // 已处理的候选样本数量\n"
                           "    float       W;      // 无偏权重（Shading 时：radiance * W = 正确估计）\n"
                           "};\n");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "更新函数（添加一个候选样本）：");
        ImGui::TextWrapped("// WRS 流算法：处理无限样本流，内存 O(1)\n"
                           "void updateReservoir(Reservoir& r, LightSample x, float w) {\n"
                           "    r.wSum += w;                        // 累加权重\n"
                           "    r.M++;\n"
                           "    if (random() < w / r.wSum)          // 概率采样（关键！）\n"
                           "        r.y = x;                        // 以概率 w/wSum 替换当前最优\n"
                           "}\n"
                           "// 数学性质：函数结束后，r.y 被选中的概率 = w_i / sum(所有w)\n"
                           "// 这正是重要性采样所需要的！\n");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "合并函数（跨像素/跨帧重用）：");
        ImGui::TextWrapped("// 合并两个 Reservoir（核心：允许跨像素/跨帧重用样本）\n"
                           "void combineReservoirs(Reservoir& dst, Reservoir src) {\n"
                           "    // src.W * src.M = src 中最优样本的\"有效权重\"\n"
                           "    updateReservoir(dst, src.y, src.W * src.M);\n"
                           "    dst.M += src.M;                     // 合并已处理样本计数\n"
                           "}\n"
                           "// 合并 N 个邻居的 Reservoir，相当于同时处理了 N 倍样本\n");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "无偏权重计算（Shading 前）：");
        ImGui::TextWrapped("// 计算输出权重 W（用于最终 Shading）\n"
                           "void finalizeReservoir(Reservoir& r, float targetPdf) {\n"
                           "    // 如果 targetPdf = 0（样本不可见），W = 0\n"
                           "    r.W = (targetPdf > 0) ? r.wSum / (targetPdf * r.M) : 0;\n"
                           "}\n"
                           "// 使用方式：\n"
                           "// radiance = evaluateRadiance(r.y) * r.W;\n"
                           "// 当 M 很大时（积累了很多帧），r.W 趋向于真实解\n");
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1, 0.85f, 0.3f, 1), "ReSTIR DI 完整流程（GPU Shader）：");
        ImGui::TextWrapped("// Pass 1：Initial Sampling（每像素处理 K 个候选光源）\n"
                           "for (int i = 0; i < K; ++i) {\n"
                           "    LightSample x = uniformSampleLight(random());\n"
                           "    float w = targetPdf(x) / sourcePdf(x);  // MIS 权重\n"
                           "    updateReservoir(reservoir[pixelId], x, w);\n"
                           "}\n\n"
                           "// Pass 2：Temporal Reuse（合并上一帧）\n"
                           "ivec2 prevCoord = reproject(pixelId, motionVector[pixelId]);\n"
                           "if (isValid(prevCoord))\n"
                           "    combineReservoirs(reservoir[pixelId], prevReservoir[prevCoord]);\n\n"
                           "// Pass 3：Spatial Reuse（合并 K 个邻居）\n"
                           "for (int k = 0; k < K_SPATIAL; ++k) {\n"
                           "    ivec2 neighbor = pixelId + randomOffset(radius);\n"
                           "    combineReservoirs(reservoir[pixelId], reservoir[neighbor]);\n"
                           "}\n\n"
                           "// Pass 4：Shading\n"
                           "finalizeReservoir(reservoir[pixelId], targetPdf(reservoir[pixelId].y));\n"
                           "color[pixelId] = shade(reservoir[pixelId].y) * reservoir[pixelId].W;\n");

        ImGui::EndTabItem();
    }

    void buildTabAlgorithm() {
        if (!ImGui::BeginTabItem("算法流程详解"))
            return;

        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1), "ReSTIR DI 每帧四个阶段");
        ImGui::Separator();
        ImGui::Spacing();

        struct Phase {
            const char* title;
            const char* detail;
            ImVec4 color;
        };
        Phase phases[] = {
            {"① Initial Sampling（初始候选采样）",
             "每像素随机选取 K=32 个光源候选。\n"
             "对每个候选计算 targetPdf（该光源对此像素的贡献估计）。\n"
             "通过 WRS 选出当前最优候选存入 Reservoir。\n"
             "代价：K 次光线求交（可见性测试），K=32 时已经相当耗时。",
             {0.4f, 0.9f, 1.0f, 1}},
            {"② Temporal Reuse（时间复用）",
             "利用 Motion Vector 将当前像素映射到上一帧的坐标。\n"
             "验证：上一帧的最优样本在当前帧是否仍然可见（1 次光线求交）。\n"
             "如果可见：合并上一帧的 Reservoir 到当前帧。\n"
             "效果：等效积累了上一帧的 M 个样本（M 随时间指数增长，通常限制 M ≤ 20×K）。",
             {0.3f, 1.0f, 0.5f, 1}},
            {"③ Spatial Reuse（空间复用）",
             "在半径 R 像素范围内随机选择 K_s=5 个邻居像素。\n"
             "验证：邻居的最优样本对当前像素是否可见（K_s 次光线求交）。\n"
             "合并验证通过的邻居 Reservoir 到当前像素。\n"
             "注意：需要 MIS 权重修正，否则会引入偏差（bias）。",
             {1.0f, 0.7f, 0.3f, 1}},
            {"④ Shading（着色）",
             "用最终 Reservoir 中的最优样本进行完整的 BRDF × 光照计算。\n"
             "用 W（无偏权重）将估计值修正为无偏结果。\n"
             "可选：经过降噪器（SVGF/ReLAX）进一步平滑。\n"
             "输出：收敛到高质量直接光照结果。",
             {0.8f, 0.5f, 1.0f, 1}},
        };
        for (const auto& p : phases) {
            ImGui::TextColored(p.color, "%s", p.title);
            ImGui::TextWrapped("%s", p.detail);
            ImGui::Spacing();
        }

        ImGui::Separator();
        ImGui::TextColored(ImVec4(1, 0.85f, 0.3f, 1), "ReSTIR 家族扩展：");
        ImGui::BulletText("ReSTIR DI（2020）：直接光照，本章演示");
        ImGui::BulletText("ReSTIR GI（2021）：间接光照（GI），将 Path 存入 Reservoir");
        ImGui::BulletText("ReSTIR PT（2022）：完整路径追踪，每个路径前缀存入 Reservoir");
        ImGui::BulletText("World-space ReSTIR（2023）：以世界空间 Irradiance Cache 为媒介");
        ImGui::BulletText("NVIDIA RTX 4090 的 Path Tracing 模式底层即使用 ReSTIR PT");

        ImGui::EndTabItem();
    }

    void buildTabSimulation() {
        if (!ImGui::BeginTabItem("CPU 端模拟演示"))
            return;

        ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1), "Mac 光栅化近似：CPU 模拟 Reservoir 收敛速度");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextWrapped("由于 macOS 不支持光线追踪，本演示在 CPU 端模拟 Reservoir 采样。\n"
                           "模拟场景：%d 个像素，每帧 %d 个随机光源，比较两种采样策略的收敛误差。",
                           pixelCount_,
                           lightCount_);
        ImGui::Spacing();

        // ── 控制面板 ──────────────────────────────────────────────────────────
        if (ImGui::SliderInt("光源数量", &lightCount_, 4, 256)) {
            simulator_ = std::make_unique<SamplingSimulator>(lightCount_, pixelCount_);
        }
        if (ImGui::SliderInt("像素数量", &pixelCount_, 16, 512)) {
            simulator_ = std::make_unique<SamplingSimulator>(lightCount_, pixelCount_);
        }
        ImGui::SliderFloat("步进间隔（秒）", &stepInterval_, 0.01f, 0.2f);

        if (ImGui::Button(isRunning_ ? "⏸ 暂停" : "▶ 开始运行")) {
            isRunning_ = !isRunning_;
            lastTime_ = std::chrono::steady_clock::now();
        }
        ImGui::SameLine();
        if (ImGui::Button("🔄 重置")) {
            simulator_->reset();
            isRunning_ = false;
            latestUniformErr_ = 0.f;
            latestRestirErr_ = 0.f;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ── 当前帧误差 ────────────────────────────────────────────────────────
        int frames = simulator_->frameCount();
        ImGui::Text("已模拟帧数：%d / %d", frames, SamplingSimulator::MAX_FRAMES);
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "均匀采样当前误差：%.4f", latestUniformErr_);
        ImGui::ProgressBar(std::min(latestUniformErr_ / 2.f, 1.f), ImVec2(-1, 16));

        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "ReSTIR    当前误差：%.4f", latestRestirErr_);
        ImGui::ProgressBar(std::min(latestRestirErr_ / 2.f, 1.f), ImVec2(-1, 16));

        if (latestUniformErr_ > 0.f && latestRestirErr_ > 0.f) {
            float ratio = latestUniformErr_ / latestRestirErr_;
            ImGui::TextColored(ImVec4(1, 0.85f, 0.3f, 1), "误差比值（均匀 / ReSTIR）：%.2fx", ratio);
        }
        ImGui::Spacing();

        // ── 折线图（误差随帧数的变化） ────────────────────────────────────────
        if (frames > 1) {
            const auto& uErr = simulator_->uniformErrors();
            const auto& rErr = simulator_->restirErrors();

            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "误差收敛曲线（帧数 → 误差，越低越好）：");

            // 归一化到相同比例
            float maxErr = 0.f;
            for (int i = 0; i < frames; ++i)
                maxErr = std::max({maxErr, uErr[size_t(i)], rErr[size_t(i)]});
            maxErr = std::max(maxErr, 0.001f);

            std::vector<float> uNorm(static_cast<size_t>(frames));
            std::vector<float> rNorm(static_cast<size_t>(frames));
            for (int i = 0; i < frames; ++i) {
                uNorm[i] = uErr[static_cast<size_t>(i)] / maxErr;
                rNorm[i] = rErr[static_cast<size_t>(i)] / maxErr;
            }

            ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "均匀采样误差");
            ImGui::PlotLines("##uniform", uNorm.data(), frames, 0, nullptr, 0.f, 1.f, ImVec2(-1, 60));

            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1), "ReSTIR 误差");
            ImGui::PlotLines("##restir", rNorm.data(), frames, 0, nullptr, 0.f, 1.f, ImVec2(-1, 60));
        } else {
            ImGui::TextDisabled("（点击[开始运行]查看收敛曲线）");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "实际 GPU 实现注意事项：");
        ImGui::BulletText("Reservoir 存储在 R32G32B32A32_FLOAT 格式的 GBuffer 中");
        ImGui::BulletText("时间重用需要 Motion Vector + 深度/法线验证（防止 temporal lag）");
        ImGui::BulletText("空间重用需要限制邻居像素的法线/深度相似性（防止漏光）");
        ImGui::BulletText("M 值需要限制上限（通常 20×K），防止时间累积过多导致无法响应变化");
        ImGui::BulletText("偏差（Bias）问题：基础 ReSTIR 有偏，需要 MIS 修正实现无偏版本");

        ImGui::EndTabItem();
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第83章：ReSTIR（Reservoir Spatio-Temporal Importance Resampling）\n";
    std::cout << " Vulkan 现代渲染技术系列 — ch83\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch83App app;
        app.run("第83章：ReSTIR 直接光照重采样", 960, 720);
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
