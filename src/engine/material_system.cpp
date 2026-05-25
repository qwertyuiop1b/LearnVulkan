/**
 * @file material_system.cpp
 * @brief 第69章：MaterialInstance + MaterialLibrary 实现
 *
 * 核心设计：
 *   Material       — 参数槽定义模板（含 Pipeline + PipelineLayout）
 *   MaterialInstance — 实例化的参数集合，持有 per-frame UBO + DescriptorSet
 *   MaterialLibrary  — 管理所有 Material 定义和 MaterialInstance，支持 PBR 预设
 */

#include <vulkan_tutorial/engine/material_system.hpp>
#include <vulkan_tutorial/utils.hpp>

#include <cstring>
#include <stdexcept>
#include <variant>

namespace engine {

// ─── Material ─────────────────────────────────────────────────────────────────

const ParamSlot* Material::findSlot(const std::string& name) const
{
    for (const auto& slot : slots)
        if (slot.name == name) return &slot;
    return nullptr;
}

// ─── MaterialInstance ─────────────────────────────────────────────────────────

uint32_t MaterialInstance::s_nextId = 0;

MaterialInstance::MaterialInstance(const Material* mat, uint32_t frameCount)
    : mat_(mat)
    , id_(s_nextId++)
    , frameCount_(frameCount)
{
    descSets_.resize(frameCount, VK_NULL_HANDLE);
    ubos_.resize(frameCount);
    uboMapped_.resize(frameCount, nullptr);

    // 用 Material 的 default value 初始化参数表
    for (const auto& slot : mat->slots)
        params_[slot.name] = slot.defaultValue;
}

MaterialInstance::~MaterialInstance()
{
    for (auto& ubo : ubos_)
        ubo.destroy();
}

MaterialInstance& MaterialInstance::set(const std::string& name, const ParamValue& value)
{
    params_[name] = value;
    return *this;
}

MaterialInstance& MaterialInstance::set(const std::string& name, Texture* tex)
{
    params_[name] = tex;
    return *this;
}

const ParamValue* MaterialInstance::get(const std::string& name) const
{
    auto it = params_.find(name);
    return it != params_.end() ? &it->second : nullptr;
}

void MaterialInstance::updateDescriptors(RHIDevice& dev,
                                          DescriptorAllocator& alloc,
                                          DescriptorLayoutCache& cache,
                                          uint32_t frameIndex)
{
    // 延迟创建 per-frame UBO（首次调用时分配 GPU 内存）
    if (!ubos_[frameIndex].isValid()) {
        Buffer::CreateInfo ci{};
        ci.size          = sizeof(PBRParams);
        ci.usage         = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        ci.memProps      = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                         | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        ci.persistentMap = true;
        ubos_[frameIndex].create(dev, ci);
        uboMapped_[frameIndex] = ubos_[frameIndex].mapped();
    }

    // 将标量 / vec4 参数打包进 PBRParams
    PBRParams packed{};

    auto getVec4 = [&](const std::string& name, glm::vec4 def) -> glm::vec4 {
        auto it = params_.find(name);
        if (it == params_.end()) return def;
        if (auto* v = std::get_if<glm::vec4>(&it->second)) return *v;
        if (auto* f = std::get_if<float>(&it->second)) return glm::vec4(*f);
        return def;
    };
    auto getFloat = [&](const std::string& name, float def) -> float {
        auto it = params_.find(name);
        if (it == params_.end()) return def;
        if (auto* f = std::get_if<float>(&it->second)) return *f;
        if (auto* v = std::get_if<glm::vec4>(&it->second)) return v->x;
        return def;
    };

    packed.albedo        = getVec4("albedo", glm::vec4(1));
    packed.metallic      = getFloat("metallic", 0.0f);
    packed.roughness     = getFloat("roughness", 0.5f);
    packed.ao            = getFloat("ao", 1.0f);
    packed.emissive      = getFloat("emissive", 0.0f);
    packed.emissiveColor = getVec4("emissiveColor", glm::vec4(0));

    if (uboMapped_[frameIndex])
        std::memcpy(uboMapped_[frameIndex], &packed, sizeof(PBRParams));

    // 构建 DescriptorSet：binding 0 = UBO，binding 1/2 = 纹理槽
    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = ubos_[frameIndex].handle();
    uboInfo.offset = 0;
    uboInfo.range  = sizeof(PBRParams);

    DescriptorBuilder builder(alloc, cache);
    builder.bindBuffer(0, uboInfo,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    // 绑定纹理槽（按槽定义顺序）
    for (const auto& slot : mat_->slots) {
        if (slot.descType != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) continue;
        auto it = params_.find(slot.name);
        if (it == params_.end()) continue;
        auto* tex = std::get_if<Texture*>(&it->second);
        if (!tex || !*tex) continue;

        VkDescriptorImageInfo imgInfo = (*tex)->descriptorInfo();
        builder.bindImage(slot.binding, imgInfo,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    VkDescriptorSetLayout layout;
    builder.build(descSets_[frameIndex], layout);
}

void MaterialInstance::bind(VkCommandBuffer cmd,
                             uint32_t frameIndex,
                             uint32_t firstSet) const
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mat_->pipeline);

    if (descSets_[frameIndex] != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            mat_->pipelineLayout,
            firstSet, 1, &descSets_[frameIndex],
            0, nullptr);
    }
}

// ─── MaterialLibrary ──────────────────────────────────────────────────────────

void MaterialLibrary::init(RHIDevice& dev, uint32_t frameCount)
{
    dev_        = &dev;
    frameCount_ = frameCount;
    allocator_.init(dev);
    layoutCache_.init(dev);
}

void MaterialLibrary::destroy()
{
    instances_.clear();
    materials_.clear();
    allocator_.destroy();
    layoutCache_.destroy();
}

void MaterialLibrary::registerMaterial(std::unique_ptr<Material> mat)
{
    materials_[mat->name] = std::move(mat);
}

void MaterialLibrary::registerPBR(const std::string& name,
                                   VkPipeline pipeline,
                                   VkPipelineLayout pipelineLayout)
{
    auto mat           = std::make_unique<Material>();
    mat->name          = name;
    mat->pipeline      = pipeline;
    mat->pipelineLayout = pipelineLayout;

    // slot 0: UBO（PBRParams）— binding 0
    mat->addSlot({ "ubo",          0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                   PBRParams{}.albedo });

    // 参数槽（对应 PBRParams 字段）
    mat->addSlot({ "albedo",       0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                   glm::vec4(1.0f) });
    mat->addSlot({ "metallic",     0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                   0.0f });
    mat->addSlot({ "roughness",    0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                   0.5f });
    mat->addSlot({ "ao",           0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                   1.0f });
    mat->addSlot({ "emissive",     0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                   0.0f });
    mat->addSlot({ "emissiveColor",0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                   glm::vec4(0.0f) });

    // 纹理槽 — binding 1/2
    mat->addSlot({ "albedoTex",    1, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   static_cast<Texture*>(nullptr) });
    mat->addSlot({ "normalTex",    2, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   static_cast<Texture*>(nullptr) });

    materials_[name] = std::move(mat);
}

MaterialInstance* MaterialLibrary::instantiate(const std::string& materialName)
{
    auto it = materials_.find(materialName);
    if (it == materials_.end())
        throw std::runtime_error("MaterialLibrary: 未找到材质 " + materialName);

    auto inst = std::make_unique<MaterialInstance>(it->second.get(), frameCount_);
    MaterialInstance* ptr = inst.get();
    instances_.push_back(std::move(inst));
    return ptr;
}

const Material* MaterialLibrary::getMaterial(const std::string& name) const
{
    auto it = materials_.find(name);
    return it != materials_.end() ? it->second.get() : nullptr;
}

void MaterialLibrary::updateAll(uint32_t frameIndex)
{
    for (auto& inst : instances_)
        inst->updateDescriptors(*dev_, allocator_, layoutCache_, frameIndex);
}

} // namespace engine
