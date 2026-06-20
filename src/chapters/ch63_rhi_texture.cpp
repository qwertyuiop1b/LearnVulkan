/**
 * @file ch63_rhi_texture.cpp
 * @brief 第63章：纹理系统（Texture2D / RenderTarget / TextureCache）
 *
 * 【封装层次】
 *   Texture2D   — loadFromFile() 自动处理 stb_image + staging + mip 生成
 *   RenderTarget — 可作为 Attachment 和 Sampled Texture 的双用途纹理
 *   TextureCache — 路径 → Texture2D 缓存，isCached() 检测，避免重复分配显存
 *
 * 【核心 API】
 *   TextureCache cache;
 *   cache.init(dev);
 *   Texture2D& brick = cache.load("assets/textures/brick_diffuse.png");
 *   Texture2D& brick2= cache.load("assets/textures/brick_diffuse.png"); // 命中缓存！
 *   assert(&brick == &brick2);
 *   cache.stats().cacheHits  // == 1
 *   cache.stats().cacheMisses// == 1
 */

#include <vulkan_tutorial/engine/demo_app.hpp>
#include <vulkan_tutorial/engine/rhi_texture.hpp>
#include <vulkan_tutorial/asset_path.hpp>

#include <filesystem>
#include <sstream>
#include <vector>

class Ch63App : public DemoApp {
  protected:
    void onInit() override {
        bgColor_ = {0.05f, 0.08f, 0.06f};
        // 模拟 TextureCache 状态
        simulateCache();
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第63章：纹理系统");
        ImGui::Separator();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(940, 690), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Texture 封装 — TextureCache / Texture2D / RenderTarget", nullptr)) {
            ImGui::End();
            return;
        }

        if (ImGui::BeginTabBar("TexTabs")) {

            if (ImGui::BeginTabItem("TextureCache API")) {
                ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "TextureCache — 防止重复加载同一纹理");
                ImGui::Separator();
                ImGui::TextWrapped("// 传统做法：每次手动检查是否已加载\n"
                                   "// if (loadedTextures.find(path) != loadedTextures.end())\n"
                                   "//     return loadedTextures[path];  // 自己维护 map\n\n"
                                   "// TextureCache 封装：\n"
                                   "TextureCache cache;\n"
                                   "cache.init(dev);\n\n"
                                   "// 第一次加载：从磁盘读取，上传 GPU\n"
                                   "Texture2D& t1 = cache.load(\"brick_diffuse.png\");\n\n"
                                   "// 再次加载相同路径：直接返回已有 Texture2D，不重新分配！\n"
                                   "Texture2D& t2 = cache.load(\"brick_diffuse.png\");\n"
                                   "assert(&t1 == &t2);  // 同一对象\n\n"
                                   "// 统计：\n"
                                   "cache.stats().totalLoaded  // 实际加载次数\n"
                                   "cache.stats().cacheHits    // 命中次数（无重复分配）\n"
                                   "cache.stats().cacheMisses  // 未命中次数（真实加载）\n\n"
                                   "// 卸载：\n"
                                   "cache.unload(\"brick_diffuse.png\");  // 释放 GPU 内存\n"
                                   "cache.unloadAll();                    // 清空缓存\n");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Texture2D API")) {
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "Texture2D — 从文件加载，自动生成 Mip");
                ImGui::Separator();
                ImGui::TextWrapped("// 传统写法（约 80 行）：\n"
                                   "//   stbi_uc* pixels = stbi_load(path, &w, &h, &ch, 4);\n"
                                   "//   VkBuffer staging; VkDeviceMemory stagingMem;\n"
                                   "//   createBuffer(staging, stagingMem, ...);\n"
                                   "//   vkMapMemory / memcpy / vkUnmapMemory;\n"
                                   "//   VkImage image; VkDeviceMemory imageMem;\n"
                                   "//   createImage(image, imageMem, ...);\n"
                                   "//   transitionLayout(UNDEFINED → TRANSFER_DST);\n"
                                   "//   vkCmdCopyBufferToImage(...);\n"
                                   "//   for (uint32_t mip = 1; mip < mipLevels; ++mip) { ... blit ... }\n"
                                   "//   createImageView(...);\n"
                                   "//   createSampler(...);\n\n"
                                   "// Texture2D 封装：\n"
                                   "Texture2D tex;\n"
                                   "tex.loadFromFile(dev,\n"
                                   "    \"assets/textures/brick_diffuse.png\",\n"
                                   "    true,   // sRGB 颜色空间\n"
                                   "    true    // 自动生成 Mip\n"
                                   ");\n\n"
                                   "// 用于描述符写入：\n"
                                   "auto info = tex.descriptorInfo();\n"
                                   "// info.sampler / info.imageView / info.imageLayout 都已准备好\n\n"
                                   "// 属性查询：\n"
                                   "tex.width()      // 图像宽度\n"
                                   "tex.height()     // 图像高度\n"
                                   "tex.mipLevels()  // Mip 层级数\n"
                                   "tex.format()     // VkFormat\n");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("RenderTarget API")) {
                ImGui::TextColored(ImVec4(0.8f, 0.5f, 1, 1), "RenderTarget — 既可作 Attachment 又可采样");
                ImGui::Separator();
                ImGui::TextWrapped("// 应用场景：shadow map / SSAO / bloom / 反射 RTT\n\n"
                                   "RenderTarget shadowMap;\n"
                                   "shadowMap.create(dev, {\n"
                                   "    .width         = 2048,\n"
                                   "    .height        = 2048,\n"
                                   "    .type          = RenderTarget::Type::Depth,\n"
                                   "    .format        = VK_FORMAT_UNDEFINED,  // 自动选最佳深度格式\n"
                                   "    .needSampling  = true,   // 用于着色器采样\n"
                                   "    .sampler       = { .magFilter = LINEAR, .addressMode = CLAMP },\n"
                                   "});\n\n"
                                   "// 作为 Framebuffer Attachment：\n"
                                   "shadowMap.view()   // VkImageView\n\n"
                                   "// 作为着色器采样：\n"
                                   "auto info = shadowMap.descriptorInfo();\n"
                                   "// 自动设置 layout = DEPTH_STENCIL_READ_ONLY_OPTIMAL\n\n"
                                   "RenderTarget bloomTex;\n"
                                   "bloomTex.create(dev, {\n"
                                   "    .type = RenderTarget::Type::Color,\n"
                                   "    .format = VK_FORMAT_R16G16B16A16_SFLOAT,\n"
                                   "    .needSampling = true,\n"
                                   "});\n");
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("缓存统计")) {
                ImGui::TextColored(ImVec4(0.4f, 1, 0.8f, 1), "模拟 TextureCache 运行统计");
                ImGui::Separator();
                ImGui::Text("已加载纹理数量   : %zu", cachedPaths_.size());
                ImGui::Text("总加载次数       : %d", totalLoads_);
                ImGui::Text("缓存命中次数     : %d", cacheHits_);
                ImGui::Text("缓存未命中次数   : %d", cacheMisses_);
                float hitRate =
                    (cacheHits_ + cacheMisses_) > 0 ? float(cacheHits_) / (cacheHits_ + cacheMisses_) * 100.0f : 0.0f;
                ImGui::Text("命中率           : %.1f %%", hitRate);
                ImGui::Separator();
                ImGui::Text("已缓存纹理列表：");
                for (const auto& p : cachedPaths_)
                    ImGui::BulletText("%s", p.c_str());
                ImGui::Spacing();
                if (ImGui::Button("模拟重复加载（+命中）")) {
                    ++cacheHits_;
                    ++totalLoads_;
                }
                ImGui::SameLine();
                if (ImGui::Button("模拟新纹理加载（+未命中）")) {
                    ++cacheMisses_;
                    ++totalLoads_;
                    cachedPaths_.push_back("texture_" + std::to_string(totalLoads_) + ".png");
                }
                ImGui::SameLine();
                if (ImGui::Button("卸载所有")) {
                    cachedPaths_.clear();
                    cacheHits_ = 0;
                    cacheMisses_ = 0;
                    totalLoads_ = 0;
                }
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

  private:
    std::vector<std::string> cachedPaths_;
    int totalLoads_ = 0, cacheHits_ = 0, cacheMisses_ = 0;

    void simulateCache() {
        cachedPaths_ = {"brick_diffuse.png", "brick_normal.png", "metal_albedo.png"};
        totalLoads_ = 8;
        cacheHits_ = 5;
        cacheMisses_ = 3;
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第63章：纹理系统（Texture2D / TextureCache）\n";
    std::cout << " 引擎封装系列 — ch63/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    try {
        Ch63App app;
        app.run("第63章：纹理系统");
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
