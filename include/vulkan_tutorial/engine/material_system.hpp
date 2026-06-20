#pragma once
/**
 * @file material_system.hpp
 * @brief 第69章：材质系统
 *
 * 层次：
 *   MaterialParam   — 类型安全的参数值（float/vec4/texture/bool）
 *   Material        — ShaderProgram 引用 + 参数定义（模板）
 *   MaterialInstance — 继承 Material，覆盖部分参数值
 *   MaterialLibrary  — 按名字管理所有材质，支持 JSON 描述
 *
 * 核心思想（对比"硬编码"写法）：
 *   传统写法：为每种材质手工写 descriptor set 更新代码
 *   材质系统：材质定义参数槽，运行时绑定具体值，统一 updateDescriptors()
 */

#include "rhi_device.hpp"
#include "rhi_texture.hpp"
#include "rhi_buffer.hpp"
#include "descriptor_manager.hpp"
#include <glm/glm.hpp>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace engine {

// ─── 参数值类型 ──────────────────────────────────────────────────────────

using ParamValue = std::variant<float,
                                glm::vec2,
                                glm::vec4,
                                int32_t,
                                glm::mat4,
                                Texture* // 纹理槽（弱引用，由 TextureCache 持有生命周期）
                                >;

// ─── 参数槽定义 ──────────────────────────────────────────────────────────

struct ParamSlot {
    std::string name;
    uint32_t binding = 0;
    uint32_t set = 0;
    VkDescriptorType descType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ParamValue defaultValue; ///< 默认值（MaterialInstance 未覆盖时用这个）
};

// ─── Material ─────────────────────────────────────────────────────────────

/**
 * @brief 材质模板（定义参数槽和 Shader）
 *
 * 一个 Material 对应一个 ShaderProgram + 一组参数槽定义。
 * 不直接持有 GPU 数据（由 MaterialInstance 持有）。
 */
class Material {
  public:
    std::string name;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    std::vector<ParamSlot> slots;

    void addSlot(const ParamSlot& slot) {
        slots.push_back(slot);
    }
    [[nodiscard]] const ParamSlot* findSlot(const std::string& name) const;
};

// ─── MaterialInstance ─────────────────────────────────────────────────────

/**
 * @brief 材质实例 —— 一个 Material 的具体参数化
 *
 * 每个 MaterialInstance：
 *   - 共享父 Material 的 Pipeline / PipelineLayout
 *   - 持有自己的 VkDescriptorSet（每帧版本，用于 UBO 更新）
 *   - 参数值覆盖（未设置的槽使用 Material 的默认值）
 *
 * 使用示例：
 * @code
 *   auto inst = lib.instantiate("pbr");
 *   inst->set("albedo",   glm::vec4(1, 0, 0, 1));
 *   inst->set("metallic", 0.8f);
 *   inst->set("albedoTex", &cache.load("rock.png"));
 *   inst->uploadUBO(fi);
 *   inst->bind(cmd, fi);
 * @endcode
 */
class MaterialInstance {
  public:
    MaterialInstance(const Material* mat, uint32_t frameCount);
    ~MaterialInstance();

    /// 设置参数（按名字）
    MaterialInstance& set(const std::string& name, const ParamValue& value);
    MaterialInstance& set(const std::string& name, Texture* tex);

    [[nodiscard]] const ParamValue* get(const std::string& name) const;

    /// 更新 UBO + 写 Descriptor
    void
    updateDescriptors(RHIDevice& dev, DescriptorAllocator& alloc, DescriptorLayoutCache& cache, uint32_t frameIndex);

    /// 绑定到命令缓冲区（设置管线 + 描述符集）
    void bind(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t firstSet = 0) const;

    [[nodiscard]] const Material* material() const {
        return mat_;
    }
    [[nodiscard]] uint32_t id() const {
        return id_;
    }

    void destroyBuffers(); ///< 由 MaterialLibrary 调用

  private:
    friend class MaterialLibrary;

    const Material* mat_ = nullptr;
    uint32_t id_ = 0;
    uint32_t frameCount_ = 2;

    std::unordered_map<std::string, ParamValue> params_; // 覆盖值

    // GPU 端：UBO（向量 / 矩阵参数打包后上传）
    std::vector<Buffer> ubos_; // per-frame
    std::vector<void*> uboMapped_;
    std::vector<VkDescriptorSet> descSets_; // per-frame

    static uint32_t s_nextId;
};

// ─── PBR 材质工具 ─────────────────────────────────────────────────────────

/// 标准 PBR 参数（对齐到 GPU std140）
struct PBRParams {
    alignas(16) glm::vec4 albedo{1, 1, 1, 1};
    alignas(4) float metallic = 0.0f;
    alignas(4) float roughness = 0.5f;
    alignas(4) float ao = 1.0f;
    alignas(4) float emissive = 0.0f;
    alignas(16) glm::vec4 emissiveColor{0, 0, 0, 0};
};

// ─── MaterialLibrary ──────────────────────────────────────────────────────

/**
 * @brief 材质库 —— 管理所有 Material 定义和 MaterialInstance
 *
 * 支持：
 *   - 代码注册（registerMaterial）
 *   - 未来可扩展为 JSON 描述文件加载
 *
 * 使用示例：
 * @code
 *   MaterialLibrary lib;
 *   lib.init(dev, frameCount);
 *
 *   // 注册 PBR 材质模板（只需注册一次）
 *   lib.registerPBR("pbr_opaque", pipeline, pipelineLayout);
 *
 *   // 创建实例（可创建无数个，共享模板）
 *   auto* inst = lib.instantiate("pbr_opaque");
 *   inst->set("albedo", glm::vec4(0.8f, 0.2f, 0.2f, 1));
 *   inst->set("metallic", 0.9f);
 *
 *   // 渲染时：
 *   inst->bind(cmd, frameIndex);
 *   mesh.drawIndexed(cmd);
 * @endcode
 */
class MaterialLibrary {
  public:
    void init(RHIDevice& dev, uint32_t frameCount);
    void destroy();

    /// 注册材质定义
    void registerMaterial(std::unique_ptr<Material> mat);

    /// 注册标准 PBR 材质（自动设置参数槽）
    void registerPBR(const std::string& name, VkPipeline pipeline, VkPipelineLayout pipelineLayout);

    /// 根据材质名字创建实例
    [[nodiscard]] MaterialInstance* instantiate(const std::string& materialName);

    /// 获取已注册的材质定义
    [[nodiscard]] const Material* getMaterial(const std::string& name) const;

    /// 每帧更新所有 dirty 的 instance 的描述符
    void updateAll(uint32_t frameIndex);

    [[nodiscard]] size_t materialCount() const {
        return materials_.size();
    }
    [[nodiscard]] size_t instanceCount() const {
        return instances_.size();
    }

  private:
    RHIDevice* dev_ = nullptr;
    uint32_t frameCount_ = 2;

    std::unordered_map<std::string, std::unique_ptr<Material>> materials_;
    std::vector<std::unique_ptr<MaterialInstance>> instances_;

    DescriptorAllocator allocator_;
    DescriptorLayoutCache layoutCache_;
};

} // namespace engine
