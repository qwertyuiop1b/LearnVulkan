/**
 * @file rhi_shader.cpp
 * @brief 第64章：着色器系统实现
 *
 * 实现内容：
 *   Shader        — 从 .spv 文件加载 + 简化版 SPIR-V 反射
 *   ShaderProgram — 多阶段组合，自动合并 DescriptorSetLayout
 *   ShaderLibrary — 按名字管理着色器程序，支持文件时间戳热重载
 *
 * SPIR-V 反射策略（不依赖 spirv-reflect 第三方库）：
 *   扫描字节码中的 OpDecorate 指令：
 *     - Decoration 33 → Binding
 *     - Decoration 34 → DescriptorSet
 *   扫描 OpVariable 获取存储类（StorageClass）：
 *     - 0 = UniformConstant → CombinedImageSampler
 *     - 2 = Uniform         → UniformBuffer
 *     - 12 = StorageBuffer  → StorageBuffer
 */

#include <vulkan_tutorial/engine/rhi_shader.hpp>
#include <vulkan_tutorial/utils.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace engine {

// ─── Shader：Move 语义 ───────────────────────────────────────────────────────

Shader::Shader(Shader&& o) noexcept
    : dev_(o.dev_), module_(o.module_), stage_(o.stage_),
      bindings_(std::move(o.bindings_)), pushConsts_(std::move(o.pushConsts_))
{
    o.dev_    = nullptr;
    o.module_ = VK_NULL_HANDLE;
}

Shader& Shader::operator=(Shader&& o) noexcept
{
    if (this != &o) {
        destroy();
        dev_       = o.dev_;
        module_    = o.module_;
        stage_     = o.stage_;
        bindings_  = std::move(o.bindings_);
        pushConsts_ = std::move(o.pushConsts_);
        o.dev_    = nullptr;
        o.module_ = VK_NULL_HANDLE;
    }
    return *this;
}

// ─── Shader::destroy ────────────────────────────────────────────────────────

void Shader::destroy()
{
    if (dev_ && module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(dev_->device(), module_, nullptr);
        module_ = VK_NULL_HANDLE;
    }
    dev_ = nullptr;
    bindings_.clear();
    pushConsts_.clear();
}

// ─── Shader::loadFromFile ────────────────────────────────────────────────────

void Shader::loadFromFile(RHIDevice& dev, const std::string& spvPath,
                           VkShaderStageFlagBits stage)
{
    auto code = readFile(spvPath);
    if (code.size() % 4 != 0)
        throw std::runtime_error("[Shader] SPIR-V 文件大小无效: " + spvPath);

    std::vector<uint32_t> spirv(code.size() / 4);
    std::memcpy(spirv.data(), code.data(), code.size());
    loadFromMemory(dev, spirv, stage);
}

// ─── Shader::loadFromMemory ──────────────────────────────────────────────────

void Shader::loadFromMemory(RHIDevice& dev, const std::vector<uint32_t>& spirv,
                             VkShaderStageFlagBits stage)
{
    destroy();
    dev_   = &dev;
    stage_ = stage;

    VkShaderModuleCreateInfo moduleCI{};
    moduleCI.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleCI.codeSize = spirv.size() * sizeof(uint32_t);
    moduleCI.pCode    = spirv.data();
    VK_CHECK(vkCreateShaderModule(dev.device(), &moduleCI, nullptr, &module_));

    reflect(spirv);
}

// ─── Shader::reflect ────────────────────────────────────────────────────────
//
// 简化版 SPIR-V 反射：线性扫描字节码，提取描述符绑定信息
// SPIR-V 格式：Magic(0x07230203) | Version | Generator | Bound | 0
// 每条指令：lower-16=opcode, upper-16=wordCount

void Shader::reflect(const std::vector<uint32_t>& spirv)
{
    bindings_.clear();
    pushConsts_.clear();

    if (spirv.size() < 5) return;
    if (spirv[0] != 0x07230203) return; // 魔数检查

    // 收集 id → set / binding 的映射
    std::unordered_map<uint32_t, uint32_t> setMap;
    std::unordered_map<uint32_t, uint32_t> bindingMap;
    // id → storage class（用于判断描述符类型）
    std::unordered_map<uint32_t, uint32_t> storageClassMap;
    // id → name（可选，用于调试）
    std::unordered_map<uint32_t, std::string> nameMap;

    uint32_t i = 5; // 跳过文件头
    while (i < spirv.size()) {
        const uint32_t word      = spirv[i];
        const uint32_t opcode    = word & 0xFFFFu;
        const uint32_t wordCount = (word >> 16) & 0xFFFFu;

        if (wordCount == 0 || i + wordCount > spirv.size()) break;

        switch (opcode) {
        case 5: // OpName
            if (wordCount >= 3) {
                const uint32_t id = spirv[i + 1];
                std::string name;
                for (uint32_t k = i + 2; k < i + wordCount; ++k) {
                    uint32_t w = spirv[k];
                    for (int b = 0; b < 4; ++b) {
                        char c = static_cast<char>((w >> (b * 8)) & 0xFF);
                        if (c == 0) goto nameDone;
                        name += c;
                    }
                }
                nameDone:
                nameMap[id] = name;
            }
            break;

        case 59: // OpVariable: result_type, result_id, storage_class[, initializer]
            if (wordCount >= 4) {
                const uint32_t resultId    = spirv[i + 2];
                const uint32_t storageClass = spirv[i + 3];
                storageClassMap[resultId] = storageClass;
            }
            break;

        case 71: // OpDecorate: target_id, decoration[, value]
            if (wordCount >= 3) {
                const uint32_t id         = spirv[i + 1];
                const uint32_t decoration = spirv[i + 2];
                if (decoration == 33 && wordCount >= 4) // Binding
                    bindingMap[id] = spirv[i + 3];
                if (decoration == 34 && wordCount >= 4) // DescriptorSet
                    setMap[id] = spirv[i + 3];
            }
            break;

        default:
            break;
        }

        i += wordCount;
    }

    // 合并：只保留同时有 set 和 binding 装饰的变量
    for (auto& [id, binding] : bindingMap) {
        auto setIt = setMap.find(id);
        if (setIt == setMap.end()) continue;

        BindingReflect br{};
        br.set     = setIt->second;
        br.binding = binding;
        br.stages  = static_cast<VkShaderStageFlags>(stage_);
        br.count   = 1;

        // 根据 storage class 判断描述符类型
        auto scIt = storageClassMap.find(id);
        const uint32_t sc = (scIt != storageClassMap.end()) ? scIt->second : 0u;
        if (sc == 0)       // UniformConstant → 采样器或图像
            br.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        else if (sc == 2)  // Uniform → UBO
            br.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (sc == 12) // StorageBuffer → SSBO
            br.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        else
            br.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

        auto nameIt = nameMap.find(id);
        if (nameIt != nameMap.end())
            br.name = nameIt->second;

        bindings_.push_back(br);
    }

    // 按 set + binding 排序，方便后续合并
    std::sort(bindings_.begin(), bindings_.end(),
              [](const BindingReflect& a, const BindingReflect& b) {
                  if (a.set != b.set) return a.set < b.set;
                  return a.binding < b.binding;
              });
}

// ─── Shader::stageInfo ──────────────────────────────────────────────────────

VkPipelineShaderStageCreateInfo Shader::stageInfo() const
{
    VkPipelineShaderStageCreateInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage  = stage_;
    info.module = module_;
    info.pName  = "main";
    return info;
}

// ─── ShaderProgram ───────────────────────────────────────────────────────────

void ShaderProgram::addStage(RHIDevice& dev, const std::string& spvPath,
                              VkShaderStageFlagBits stage)
{
    Shader sh;
    sh.loadFromFile(dev, spvPath, stage);
    addStage(std::move(sh));
}

void ShaderProgram::addStage(Shader&& shader)
{
    stageInfos_.push_back(shader.stageInfo());
    stages_.push_back(std::move(shader));
    // 将新阶段的反射信息追加到合并列表（link 时再去重整理）
}

// ─── ShaderProgram::link ────────────────────────────────────────────────────

void ShaderProgram::link(RHIDevice& dev)
{
    dev_ = &dev;
    mergeBindings();
    buildLayouts(dev);
}

void ShaderProgram::mergeBindings()
{
    mergedBindings_.clear();
    mergedPushConsts_.clear();

    // 合并所有阶段的 bindings，相同 (set, binding) 的合并 stages 字段
    for (const auto& stage : stages_) {
        for (const auto& b : stage.bindings()) {
            auto it = std::find_if(mergedBindings_.begin(), mergedBindings_.end(),
                [&](const BindingReflect& m) {
                    return m.set == b.set && m.binding == b.binding;
                });
            if (it != mergedBindings_.end())
                it->stages |= b.stages;
            else
                mergedBindings_.push_back(b);
        }
        for (const auto& pc : stage.pushConstants()) {
            auto it = std::find_if(mergedPushConsts_.begin(), mergedPushConsts_.end(),
                [&](const PushConstantReflect& m) { return m.offset == pc.offset; });
            if (it != mergedPushConsts_.end())
                it->stages |= pc.stages;
            else
                mergedPushConsts_.push_back(pc);
        }
    }

    std::sort(mergedBindings_.begin(), mergedBindings_.end(),
              [](const BindingReflect& a, const BindingReflect& b) {
                  if (a.set != b.set) return a.set < b.set;
                  return a.binding < b.binding;
              });
}

void ShaderProgram::buildLayouts(RHIDevice& dev)
{
    // 确定需要多少个 DescriptorSet（取最大 set + 1）
    uint32_t maxSet = 0;
    for (const auto& b : mergedBindings_)
        maxSet = std::max(maxSet, b.set);

    const uint32_t setCount = mergedBindings_.empty() ? 0 : maxSet + 1;

    // 为每个 set 收集 bindings
    setLayouts_.resize(setCount);
    for (uint32_t s = 0; s < setCount; ++s) {
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        for (const auto& b : mergedBindings_) {
            if (b.set != s) continue;
            VkDescriptorSetLayoutBinding lb{};
            lb.binding            = b.binding;
            lb.descriptorType     = b.type;
            lb.descriptorCount    = b.count;
            lb.stageFlags         = b.stages;
            lb.pImmutableSamplers = nullptr;
            layoutBindings.push_back(lb);
        }

        VkDescriptorSetLayoutCreateInfo setCI{};
        setCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setCI.bindingCount = static_cast<uint32_t>(layoutBindings.size());
        setCI.pBindings    = layoutBindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(dev.device(), &setCI, nullptr, &setLayouts_[s]));
    }

    // Push constants
    std::vector<VkPushConstantRange> pushRanges;
    for (const auto& pc : mergedPushConsts_) {
        VkPushConstantRange r{};
        r.stageFlags = pc.stages;
        r.offset     = pc.offset;
        r.size       = pc.size;
        pushRanges.push_back(r);
    }

    VkPipelineLayoutCreateInfo layoutCI{};
    layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount         = static_cast<uint32_t>(setLayouts_.size());
    layoutCI.pSetLayouts            = setLayouts_.empty() ? nullptr : setLayouts_.data();
    layoutCI.pushConstantRangeCount = static_cast<uint32_t>(pushRanges.size());
    layoutCI.pPushConstantRanges    = pushRanges.empty() ? nullptr : pushRanges.data();
    VK_CHECK(vkCreatePipelineLayout(dev.device(), &layoutCI, nullptr, &layout_));
}

void ShaderProgram::destroy()
{
    if (!dev_) return;
    VkDevice d = dev_->device();
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(d, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }
    for (auto& sl : setLayouts_)
        if (sl != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(d, sl, nullptr);
    setLayouts_.clear();
    stages_.clear();
    stageInfos_.clear();
    mergedBindings_.clear();
    mergedPushConsts_.clear();
    dev_ = nullptr;
}

VkDescriptorSetLayout ShaderProgram::descriptorSetLayout(uint32_t set) const
{
    if (set < setLayouts_.size())
        return setLayouts_[set];
    return VK_NULL_HANDLE;
}

// ─── ShaderLibrary ──────────────────────────────────────────────────────────

void ShaderLibrary::init(RHIDevice& dev, const std::string& shaderDir)
{
    dev_       = &dev;
    shaderDir_ = shaderDir;
}

void ShaderLibrary::destroy()
{
    entries_.clear();
    dev_ = nullptr;
}

static int64_t getFileTimestamp(const std::string& path)
{
    std::error_code ec;
    auto ftime = std::filesystem::last_write_time(path, ec);
    if (ec) return -1;
    return ftime.time_since_epoch().count();
}

void ShaderLibrary::registerProgram(const ProgramDesc& desc)
{
    Entry entry{};
    entry.desc = desc;
    entry.prog = std::make_unique<ShaderProgram>();

    for (auto& [spvFile, stage] : desc.stages) {
        const std::string fullPath = shaderDir_ + "/" + spvFile;
        entry.prog->addStage(*dev_, fullPath, stage);
        entry.timestamps.push_back(getFileTimestamp(fullPath));
    }
    entry.prog->link(*dev_);

    entries_[desc.name] = std::move(entry);
}

void ShaderLibrary::registerProgram(const std::string& name,
                                     const std::string& vertSpv,
                                     const std::string& fragSpv)
{
    ProgramDesc desc{};
    desc.name = name;
    desc.stages.push_back({vertSpv, VK_SHADER_STAGE_VERTEX_BIT});
    desc.stages.push_back({fragSpv, VK_SHADER_STAGE_FRAGMENT_BIT});
    registerProgram(desc);
}

void ShaderLibrary::registerCompute(const std::string& name,
                                     const std::string& compSpv)
{
    ProgramDesc desc{};
    desc.name = name;
    desc.stages.push_back({compSpv, VK_SHADER_STAGE_COMPUTE_BIT});
    registerProgram(desc);
}

ShaderProgram* ShaderLibrary::get(const std::string& name)
{
    auto it = entries_.find(name);
    return (it != entries_.end()) ? it->second.prog.get() : nullptr;
}

const ShaderProgram* ShaderLibrary::get(const std::string& name) const
{
    auto it = entries_.find(name);
    return (it != entries_.end()) ? it->second.prog.get() : nullptr;
}

bool ShaderLibrary::reloadIfDirty(const std::string& name)
{
    auto it = entries_.find(name);
    if (it == entries_.end()) return false;

    Entry& entry = it->second;
    bool dirty = false;
    for (size_t i = 0; i < entry.desc.stages.size(); ++i) {
        const std::string fullPath = shaderDir_ + "/" + entry.desc.stages[i].first;
        const int64_t ts = getFileTimestamp(fullPath);
        if (ts != entry.timestamps[i]) {
            dirty = true;
            break;
        }
    }
    if (!dirty) return false;

    // 等待设备空闲后重建
    dev_->waitIdle();
    auto newProg = std::make_unique<ShaderProgram>();
    for (size_t i = 0; i < entry.desc.stages.size(); ++i) {
        const std::string fullPath = shaderDir_ + "/" + entry.desc.stages[i].first;
        const VkShaderStageFlagBits stage = entry.desc.stages[i].second;
        newProg->addStage(*dev_, fullPath, stage);
        entry.timestamps[i] = getFileTimestamp(fullPath);
    }
    newProg->link(*dev_);
    entry.prog = std::move(newProg);

    if (onReload_) onReload_(name, entry.prog.get());
    return true;
}

void ShaderLibrary::reloadAll()
{
    for (auto& [name, _] : entries_)
        reloadIfDirty(name);
}

std::vector<std::string> ShaderLibrary::programNames() const
{
    std::vector<std::string> names;
    names.reserve(entries_.size());
    for (const auto& [name, _] : entries_)
        names.push_back(name);
    std::sort(names.begin(), names.end());
    return names;
}

} // namespace engine
