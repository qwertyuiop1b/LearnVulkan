#pragma once
/**
 * @file rhi_shader.hpp
 * @brief 第64章：着色器系统
 *
 * 封装层次：
 *   Shader        — 单个着色器阶段（SPIR-V + 基本反射信息）
 *   ShaderProgram — 顶点 + 片元（或 Compute）的组合，
 *                   自动合并 DescriptorSetLayout
 *   ShaderLibrary — 按名字管理所有着色器程序，支持热重载
 *
 * 关键设计：
 *   ShaderProgram 通过读取 SPIR-V 字节码中的绑定信息
 *   自动生成 VkDescriptorSetLayoutCreateInfo，
 *   业务代码无需手工写 binding/set 声明。
 */

#include "rhi_device.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

// ─── 反射信息 ──────────────────────────────────────────────────────────────

/// 从 SPIR-V 解析出的单个绑定描述
struct BindingReflect {
    uint32_t           set     = 0;
    uint32_t           binding = 0;
    VkDescriptorType   type    = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    uint32_t           count   = 1;
    VkShaderStageFlags stages  = 0;
    std::string        name;
};

/// 从 SPIR-V 解析出的 Push Constant 块
struct PushConstantReflect {
    uint32_t           offset  = 0;
    uint32_t           size    = 0;
    VkShaderStageFlags stages  = 0;
    std::string        name;
};

// ─── 单个着色器阶段 ────────────────────────────────────────────────────────

/**
 * @brief 单个 SPIR-V 着色器模块 + 反射信息
 *
 * 反射采用简化版（不依赖 spirv-reflect 第三方库），
 * 直接解析 SPIR-V 字节码中的 OpDecorate / OpTypePointer 指令。
 */
class Shader {
public:
    Shader() = default;
    ~Shader() { destroy(); }
    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;
    Shader(Shader&&) noexcept;
    Shader& operator=(Shader&&) noexcept;

    /// 从 .spv 文件加载
    void loadFromFile(RHIDevice& dev, const std::string& spvPath,
                      VkShaderStageFlagBits stage);
    /// 从内存中的 SPIR-V 字节码加载
    void loadFromMemory(RHIDevice& dev, const std::vector<uint32_t>& spirv,
                        VkShaderStageFlagBits stage);
    void destroy();

    [[nodiscard]] VkShaderModule           module()   const { return module_; }
    [[nodiscard]] VkShaderStageFlagBits    stage()    const { return stage_; }
    [[nodiscard]] const std::vector<BindingReflect>&      bindings()      const { return bindings_; }
    [[nodiscard]] const std::vector<PushConstantReflect>& pushConstants() const { return pushConsts_; }
    [[nodiscard]] bool isValid() const { return module_ != VK_NULL_HANDLE; }

    VkPipelineShaderStageCreateInfo stageInfo() const;

private:
    void reflect(const std::vector<uint32_t>& spirv);

    RHIDevice*              dev_       = nullptr;
    VkShaderModule          module_    = VK_NULL_HANDLE;
    VkShaderStageFlagBits   stage_     = VK_SHADER_STAGE_VERTEX_BIT;
    std::vector<BindingReflect>      bindings_;
    std::vector<PushConstantReflect> pushConsts_;
};

// ─── 着色器程序（组合多个阶段）────────────────────────────────────────────

/**
 * @brief 完整的着色器程序（顶点 + 片元 / Compute）
 *
 * 自动合并所有阶段的反射信息，生成：
 *   - VkDescriptorSetLayout[]（每个 set 一个）
 *   - VkPipelineLayout
 *
 * 使用示例：
 * @code
 *   ShaderProgram prog;
 *   prog.addStage(dev, "shaders/pbr.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
 *   prog.addStage(dev, "shaders/pbr.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
 *   prog.link(dev);
 *   // prog.pipelineLayout() 可直接用于创建管线
 * @endcode
 */
class ShaderProgram {
public:
    ShaderProgram() = default;
    ~ShaderProgram() { destroy(); }
    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;

    void addStage(RHIDevice& dev, const std::string& spvPath,
                  VkShaderStageFlagBits stage);
    void addStage(Shader&& shader);

    /// 合并反射信息，创建 DescriptorSetLayout + PipelineLayout
    void link(RHIDevice& dev);
    void destroy();

    [[nodiscard]] VkPipelineLayout pipelineLayout()  const { return layout_; }
    [[nodiscard]] VkDescriptorSetLayout descriptorSetLayout(uint32_t set = 0) const;
    [[nodiscard]] uint32_t descriptorSetCount() const
    {
        return static_cast<uint32_t>(setLayouts_.size());
    }
    [[nodiscard]] const std::vector<VkPipelineShaderStageCreateInfo>& stageInfos() const
    {
        return stageInfos_;
    }
    [[nodiscard]] const std::vector<BindingReflect>& allBindings() const { return mergedBindings_; }
    [[nodiscard]] bool isLinked() const { return layout_ != VK_NULL_HANDLE; }

private:
    void mergeBindings();
    void buildLayouts(RHIDevice& dev);

    RHIDevice*  dev_  = nullptr;
    std::vector<Shader>                           stages_;
    std::vector<VkPipelineShaderStageCreateInfo>  stageInfos_;
    std::vector<BindingReflect>                   mergedBindings_;
    std::vector<PushConstantReflect>              mergedPushConsts_;
    std::vector<VkDescriptorSetLayout>            setLayouts_;
    VkPipelineLayout                              layout_ = VK_NULL_HANDLE;
};

// ─── 着色器库 ──────────────────────────────────────────────────────────────

/**
 * @brief 按名字管理所有着色器程序，支持热重载
 *
 * 使用示例：
 * @code
 *   ShaderLibrary lib;
 *   lib.init(dev, "build/shaders");
 *   lib.registerProgram("pbr", {"pbr.vert.spv", "pbr.frag.spv"});
 *   auto* prog = lib.get("pbr");
 *
 *   // 文件变化时热重载：
 *   lib.reloadIfDirty("pbr");
 * @endcode
 */
class ShaderLibrary {
public:
    void init(RHIDevice& dev, const std::string& shaderDir);
    void destroy();

    struct ProgramDesc {
        std::string name;
        std::vector<std::pair<std::string, VkShaderStageFlagBits>> stages;
        // e.g. { {"pbr.vert.spv", VERTEX}, {"pbr.frag.spv", FRAGMENT} }
    };

    void registerProgram(const ProgramDesc& desc);
    void registerProgram(const std::string& name,
                         const std::string& vertSpv,
                         const std::string& fragSpv);
    void registerCompute(const std::string& name,
                         const std::string& compSpv);

    [[nodiscard]] ShaderProgram*       get(const std::string& name);
    [[nodiscard]] const ShaderProgram* get(const std::string& name) const;

    /// 检查 .spv 文件时间戳，若有更新则重新编译
    bool reloadIfDirty(const std::string& name);
    void reloadAll();

    [[nodiscard]] std::vector<std::string> programNames() const;

    /// 热重载回调（重载完成后通知业务代码重建管线）
    using ReloadCallback = std::function<void(const std::string& name, ShaderProgram*)>;
    void setReloadCallback(ReloadCallback cb) { onReload_ = std::move(cb); }

private:
    struct Entry {
        ProgramDesc                    desc;
        std::unique_ptr<ShaderProgram> program;
        std::vector<int64_t>           timestamps;  // 每个 .spv 的最后修改时间
    };

    RHIDevice*  dev_       = nullptr;
    std::string shaderDir_;
    std::unordered_map<std::string, Entry> entries_;
    ReloadCallback onReload_;
};

} // namespace engine
