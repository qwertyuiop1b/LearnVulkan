#pragma once

/**
 * @file gltf_loader.hpp
 * @brief glTF 2.0 网格/材质/动画解析（基于 cgltf）
 */

#include <vulkan_tutorial/asset_path.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct cgltf_data;

namespace vulkan_tutorial {

struct GltfVertex {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec4 tangent{1.0f, 0.0f, 0.0f, 1.0f};
    glm::vec2 texCoord{0.0f};
    glm::uvec4 joints{0};
    glm::vec4 weights{0.0f};
};

struct GltfMaterial {
    glm::vec4 baseColorFactor{1.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    std::string baseColorTexture;
    std::string normalTexture;
    std::string metallicRoughnessTexture;
};

struct GltfMeshPrimitive {
    std::vector<GltfVertex> vertices;
    std::vector<uint32_t> indices;
    int32_t materialIndex = -1;
};

struct GltfAnimationChannel {
    uint32_t targetNode = 0;
    uint32_t samplerIndex = 0;
    int pathType = 0;
};

struct GltfAnimationSampler {
    std::vector<float> times;
    std::vector<glm::vec3> translations;
    std::vector<glm::quat> rotations;
    std::vector<glm::vec3> scales;
};

struct GltfAnimation {
    std::string name;
    float duration = 0.0f;
    std::vector<GltfAnimationChannel> channels;
    std::vector<GltfAnimationSampler> samplers;
};

struct GltfSkin {
    std::vector<glm::mat4> inverseBindMatrices;
    std::vector<uint32_t> jointNodes;
};

struct GltfScene {
    std::vector<GltfMeshPrimitive> meshes;
    std::vector<GltfMaterial> materials;
    std::vector<GltfAnimation> animations;
    std::vector<GltfSkin> skins;
    std::string baseDirectory;
};

GltfScene loadGltfScene(const std::string& relativePath);

} // namespace vulkan_tutorial
