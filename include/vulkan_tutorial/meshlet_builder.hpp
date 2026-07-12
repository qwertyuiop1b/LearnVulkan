#pragma once

#include <glm/glm.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vulkan_tutorial {

struct OfflineMeshlet {
    uint32_t firstTriangle = 0;
    uint32_t triangleCount = 0;
    glm::vec3 center{};
    float radius = 0.0f;
    glm::vec3 coneAxis{0.0f, 0.0f, 1.0f};
    float coneCutoff = -1.0f;
};

inline std::vector<OfflineMeshlet> buildOfflineMeshlets(const std::vector<glm::vec3>& vertices,
                                                         const std::vector<uint32_t>& indices,
                                                         uint32_t maxTriangles = 64) {
    std::vector<OfflineMeshlet> meshlets;
    for (uint32_t first = 0; first < indices.size() / 3; first += maxTriangles) {
        OfflineMeshlet meshlet;
        meshlet.firstTriangle = first;
        meshlet.triangleCount = std::min(maxTriangles, uint32_t(indices.size() / 3) - first);
        glm::vec3 sum(0.0f);
        uint32_t count = 0;
        for (uint32_t triangle = 0; triangle < meshlet.triangleCount; ++triangle) {
            const uint32_t base = (first + triangle) * 3;
            for (uint32_t corner = 0; corner < 3; ++corner) {
                const uint32_t index = indices[base + corner];
                if (index >= vertices.size()) continue;
                sum += vertices[index]; ++count;
            }
        }
        meshlet.center = count ? sum / static_cast<float>(count) : glm::vec3(0.0f);
        for (uint32_t triangle = 0; triangle < meshlet.triangleCount; ++triangle) {
            const uint32_t base = (first + triangle) * 3;
            for (uint32_t corner = 0; corner < 3; ++corner) {
                const uint32_t index = indices[base + corner];
                if (index < vertices.size())
                    meshlet.radius = std::max(meshlet.radius, glm::length(vertices[index] - meshlet.center));
            }
        }
        meshlets.push_back(meshlet);
    }
    return meshlets;
}

} // namespace vulkan_tutorial
