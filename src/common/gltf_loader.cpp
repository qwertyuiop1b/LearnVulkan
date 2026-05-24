#include <vulkan_tutorial/gltf_loader.hpp>

#include "../../external/cgltf/cgltf.h"

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace vulkan_tutorial {

static std::string parentDirectory(const std::string& path)
{
    const std::filesystem::path p(path);
    return p.parent_path().string();
}

static glm::vec3 readVec3(const float* data)
{
    return glm::vec3(data[0], data[1], data[2]);
}

static glm::vec2 readVec2(const float* data)
{
    return glm::vec2(data[0], data[1]);
}

static glm::vec4 readVec4(const float* data)
{
    return glm::vec4(data[0], data[1], data[2], data[3]);
}

static void appendPrimitive(const cgltf_primitive& primitive, GltfMeshPrimitive& out)
{
    cgltf_accessor* posAcc = nullptr;
    cgltf_accessor* normAcc = nullptr;
    cgltf_accessor* tanAcc = nullptr;
    cgltf_accessor* uvAcc = nullptr;
    cgltf_accessor* jointAcc = nullptr;
    cgltf_accessor* weightAcc = nullptr;
    for (size_t i = 0; i < primitive.attributes_count; ++i) {
        const cgltf_attribute& attr = primitive.attributes[i];
        switch (attr.type) {
        case cgltf_attribute_type_position: posAcc = attr.data; break;
        case cgltf_attribute_type_normal: normAcc = attr.data; break;
        case cgltf_attribute_type_tangent: tanAcc = attr.data; break;
        case cgltf_attribute_type_texcoord: uvAcc = attr.data; break;
        case cgltf_attribute_type_joints: jointAcc = attr.data; break;
        case cgltf_attribute_type_weights: weightAcc = attr.data; break;
        default: break;
        }
    }
    if (!posAcc)
        return;
    const size_t vertexCount = posAcc->count;
    out.vertices.resize(vertexCount);
    for (size_t v = 0; v < vertexCount; ++v) {
        float pos[3] = {};
        cgltf_accessor_read_float(posAcc, v, pos, 3);
        out.vertices[v].position = readVec3(pos);
        if (normAcc) {
            float norm[3] = {};
            cgltf_accessor_read_float(normAcc, v, norm, 3);
            out.vertices[v].normal = readVec3(norm);
        }
        if (tanAcc) {
            float tan[4] = {};
            cgltf_accessor_read_float(tanAcc, v, tan, 4);
            out.vertices[v].tangent = readVec4(tan);
        }
        if (uvAcc) {
            float uv[2] = {};
            cgltf_accessor_read_float(uvAcc, v, uv, 2);
            out.vertices[v].texCoord = readVec2(uv);
        }
        if (jointAcc) {
            cgltf_uint joints[4] = {};
            cgltf_accessor_read_uint(jointAcc, v, joints, 4);
            out.vertices[v].joints = glm::uvec4(joints[0], joints[1], joints[2], joints[3]);
        }
        if (weightAcc) {
            float weights[4] = {};
            cgltf_accessor_read_float(weightAcc, v, weights, 4);
            out.vertices[v].weights = readVec4(weights);
        }
    }
    if (primitive.indices) {
        out.indices.resize(primitive.indices->count);
        for (size_t i = 0; i < primitive.indices->count; ++i)
            out.indices[i] = static_cast<uint32_t>(cgltf_accessor_read_index(primitive.indices, i));
    } else {
        out.indices.resize(vertexCount);
        for (size_t i = 0; i < vertexCount; ++i)
            out.indices[i] = static_cast<uint32_t>(i);
    }
}

GltfScene loadGltfScene(const std::string& relativePath)
{
    const std::string path = resolveAssetPath(relativePath);
    cgltf_options options{};
    cgltf_data* data = nullptr;
    const cgltf_result result = cgltf_parse_file(&options, path.c_str(), &data);
    if (result != cgltf_result_success || !data)
        throw std::runtime_error("cgltf 解析失败: " + path);
    if (cgltf_load_buffers(&options, data, path.c_str()) != cgltf_result_success) {
        cgltf_free(data);
        throw std::runtime_error("cgltf 加载 buffer 失败: " + path);
    }
    GltfScene scene;
    scene.baseDirectory = parentDirectory(path);
    scene.materials.resize(data->materials_count);
    for (size_t m = 0; m < data->materials_count; ++m) {
        const cgltf_material& mat = data->materials[m];
        GltfMaterial& out = scene.materials[m];
        if (mat.has_pbr_metallic_roughness) {
            const cgltf_pbr_metallic_roughness& pbr = mat.pbr_metallic_roughness;
            out.baseColorFactor = glm::vec4(
                pbr.base_color_factor[0], pbr.base_color_factor[1],
                pbr.base_color_factor[2], pbr.base_color_factor[3]);
            out.metallicFactor = pbr.metallic_factor;
            out.roughnessFactor = pbr.roughness_factor;
            if (pbr.base_color_texture.texture && pbr.base_color_texture.texture->image)
                out.baseColorTexture = pbr.base_color_texture.texture->image->uri ?
                    pbr.base_color_texture.texture->image->uri : "";
            if (pbr.metallic_roughness_texture.texture && pbr.metallic_roughness_texture.texture->image)
                out.metallicRoughnessTexture = pbr.metallic_roughness_texture.texture->image->uri ?
                    pbr.metallic_roughness_texture.texture->image->uri : "";
        }
        if (mat.normal_texture.texture && mat.normal_texture.texture->image)
            out.normalTexture = mat.normal_texture.texture->image->uri ?
                mat.normal_texture.texture->image->uri : "";
    }
    for (size_t meshIndex = 0; meshIndex < data->meshes_count; ++meshIndex) {
        const cgltf_mesh& mesh = data->meshes[meshIndex];
        for (size_t p = 0; p < mesh.primitives_count; ++p) {
            GltfMeshPrimitive primitive;
            appendPrimitive(mesh.primitives[p], primitive);
            if (mesh.primitives[p].material)
                primitive.materialIndex = static_cast<int32_t>(
                    mesh.primitives[p].material - data->materials);
            scene.meshes.push_back(std::move(primitive));
        }
    }
    scene.skins.resize(data->skins_count);
    for (size_t s = 0; s < data->skins_count; ++s) {
        const cgltf_skin& skin = data->skins[s];
        GltfSkin& outSkin = scene.skins[s];
        outSkin.jointNodes.resize(skin.joints_count);
        for (size_t j = 0; j < skin.joints_count; ++j)
            outSkin.jointNodes[j] = static_cast<uint32_t>(skin.joints[j] - data->nodes);
        if (skin.inverse_bind_matrices) {
            outSkin.inverseBindMatrices.resize(skin.inverse_bind_matrices->count);
            for (size_t j = 0; j < skin.inverse_bind_matrices->count; ++j) {
                float matrix[16] = {};
                cgltf_accessor_read_float(skin.inverse_bind_matrices, j, matrix, 16);
                outSkin.inverseBindMatrices[j] = glm::mat4(
                    matrix[0], matrix[1], matrix[2], matrix[3],
                    matrix[4], matrix[5], matrix[6], matrix[7],
                    matrix[8], matrix[9], matrix[10], matrix[11],
                    matrix[12], matrix[13], matrix[14], matrix[15]);
            }
        }
    }
    scene.animations.resize(data->animations_count);
    for (size_t a = 0; a < data->animations_count; ++a) {
        const cgltf_animation& anim = data->animations[a];
        GltfAnimation& outAnim = scene.animations[a];
        outAnim.name = anim.name ? anim.name : "animation";
        outAnim.duration = 0.0f;
        outAnim.channels.resize(anim.channels_count);
        outAnim.samplers.resize(anim.samplers_count);
        for (size_t c = 0; c < anim.channels_count; ++c) {
            outAnim.channels[c].targetNode = static_cast<uint32_t>(
                anim.channels[c].target_node - data->nodes);
            outAnim.channels[c].samplerIndex = static_cast<uint32_t>(
                anim.channels[c].sampler - anim.samplers);
            outAnim.channels[c].pathType = static_cast<int>(anim.channels[c].target_path);
        }
        for (size_t s = 0; s < anim.samplers_count; ++s) {
            const cgltf_animation_sampler& sampler = anim.samplers[s];
            GltfAnimationSampler& outSampler = outAnim.samplers[s];
            outSampler.times.resize(sampler.input->count);
            for (size_t i = 0; i < sampler.input->count; ++i) {
                float t = 0.0f;
                cgltf_accessor_read_float(sampler.input, i, &t, 1);
                outSampler.times[i] = t;
                outAnim.duration = std::max(outAnim.duration, t);
            }
            if (sampler.interpolation == cgltf_interpolation_type_linear) {
                if (sampler.output->type == cgltf_type_vec3) {
                    outSampler.translations.resize(sampler.output->count);
                    for (size_t i = 0; i < sampler.output->count; ++i) {
                        float v[3] = {};
                        cgltf_accessor_read_float(sampler.output, i, v, 3);
                        outSampler.translations[i] = readVec3(v);
                    }
                } else if (sampler.output->type == cgltf_type_vec4) {
                    outSampler.rotations.resize(sampler.output->count);
                    for (size_t i = 0; i < sampler.output->count; ++i) {
                        float v[4] = {};
                        cgltf_accessor_read_float(sampler.output, i, v, 4);
                        outSampler.rotations[i] = glm::quat(v[3], v[0], v[1], v[2]);
                    }
                }
            }
        }
    }
    cgltf_free(data);
    return scene;
}

} // namespace vulkan_tutorial
