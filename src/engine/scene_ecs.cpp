/**
 * @file scene_ecs.cpp
 * @brief 第68章：ECS World + TransformSystem + FrustumCuller 实现
 *
 * 设计要点：
 *   - World 以密集数组 (ComponentStorage<T>) 存储组件，Cache-Friendly
 *   - TransformSystem 处理脏标记传播（两遍遍历支持一层父子层次）
 *   - FrustumCuller 使用 Gribb-Hartmann 方法提取视锥平面
 */

#include <vulkan_tutorial/engine/scene_ecs.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine {

// ─── TransformComponent ──────────────────────────────────────────────────────

glm::mat4 TransformComponent::localMatrix() const {
    glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
    glm::mat4 rx = glm::rotate(glm::mat4(1.0f), glm::radians(rotation.x), glm::vec3(1, 0, 0));
    glm::mat4 ry = glm::rotate(glm::mat4(1.0f), glm::radians(rotation.y), glm::vec3(0, 1, 0));
    glm::mat4 rz = glm::rotate(glm::mat4(1.0f), glm::radians(rotation.z), glm::vec3(0, 0, 1));
    glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
    return t * ry * rx * rz * s;
}

// ─── CameraComponent ─────────────────────────────────────────────────────────

glm::mat4 CameraComponent::projMatrix(float aspect) const {
    glm::mat4 proj = glm::perspective(glm::radians(fovDegrees), aspect, nearPlane, farPlane);
    proj[1][1] *= -1.0f; // Vulkan Y-flip
    return proj;
}

// ─── World ────────────────────────────────────────────────────────────────────

EntityID World::createEntity(const std::string& tag) {
    EntityID id = nextId_++;
    alive_.push_back(id);
    ++aliveCount_;

    if (!tag.empty()) {
        TagComponent tc;
        tc.name = tag;
        storage<TagComponent>().add(id, std::move(tc));
    }
    return id;
}

void World::destroyEntity(EntityID id) {
    auto it = std::find(alive_.begin(), alive_.end(), id);
    if (it == alive_.end())
        return;

    alive_.erase(it);
    --aliveCount_;

    storage<TransformComponent>().remove(id);
    storage<MeshComponent>().remove(id);
    storage<MaterialComponent>().remove(id);
    storage<DirectionalLightComponent>().remove(id);
    storage<PointLightComponent>().remove(id);
    storage<CameraComponent>().remove(id);
    storage<TagComponent>().remove(id);
}

bool World::isAlive(EntityID id) const {
    return std::find(alive_.begin(), alive_.end(), id) != alive_.end();
}

EntityID World::findByTag(const std::string& tag) const {
    const ComponentStorage<TagComponent>& tags = storage<TagComponent>();
    EntityID found = NULL_ENTITY;
    tags.forEach([&](EntityID id, const TagComponent& tc) {
        if (tc.name == tag)
            found = id;
    });
    return found;
}

// ─── TransformSystem ─────────────────────────────────────────────────────────

void transformSystem(World& world) {
    auto& transforms = world.storage<TransformComponent>();

    // 第一遍：更新没有父节点的变换（或父节点已是最新的）
    transforms.forEach([&](EntityID id, TransformComponent& tc) {
        if (!tc.dirty)
            return;
        if (tc.parent == NULL_ENTITY) {
            tc.world = tc.localMatrix();
            tc.dirty = false;
        }
    });

    // 第二遍：更新有父节点的子节点
    transforms.forEach([&](EntityID /*id*/, TransformComponent& tc) {
        if (!tc.dirty)
            return;
        if (tc.parent == NULL_ENTITY)
            return;
        TransformComponent* parentTc = world.get<TransformComponent>(tc.parent);
        if (!parentTc) {
            tc.world = tc.localMatrix();
        } else {
            tc.world = parentTc->world * tc.localMatrix();
        }
        tc.dirty = false;
    });
}

// ─── AABB ─────────────────────────────────────────────────────────────────────

AABB AABB::transform(const glm::mat4& m) const {
    glm::vec3 corners[8] = {
        {min.x, min.y, min.z},
        {max.x, min.y, min.z},
        {min.x, max.y, min.z},
        {max.x, max.y, min.z},
        {min.x, min.y, max.z},
        {max.x, min.y, max.z},
        {min.x, max.y, max.z},
        {max.x, max.y, max.z},
    };

    glm::vec3 newMin{std::numeric_limits<float>::max()};
    glm::vec3 newMax{std::numeric_limits<float>::lowest()};

    for (const auto& c : corners) {
        glm::vec3 tc = glm::vec3(m * glm::vec4(c, 1.0f));
        newMin = glm::min(newMin, tc);
        newMax = glm::max(newMax, tc);
    }
    return AABB{newMin, newMax};
}

// ─── Frustum ─────────────────────────────────────────────────────────────────

Frustum Frustum::fromViewProj(const glm::mat4& vp) {
    Frustum f;
    // Gribb-Hartmann 方法：直接从 VP 矩阵行提取六个平面
    // 每个平面 = row3 ± row(i)，法线朝内

    // Left:   col3 + col0
    f.planes[0] = glm::vec4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]);

    // Right:  col3 - col0
    f.planes[1] = glm::vec4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]);

    // Bottom: col3 + col1
    f.planes[2] = glm::vec4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]);

    // Top:    col3 - col1
    f.planes[3] = glm::vec4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]);

    // Near:   col3 + col2
    f.planes[4] = glm::vec4(vp[0][3] + vp[0][2], vp[1][3] + vp[1][2], vp[2][3] + vp[2][2], vp[3][3] + vp[3][2]);

    // Far:    col3 - col2
    f.planes[5] = glm::vec4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]);

    // 归一化（让 D 分量含义一致）
    for (auto& plane : f.planes) {
        float len = glm::length(glm::vec3(plane));
        if (len > 1e-6f)
            plane /= len;
    }
    return f;
}

bool Frustum::intersects(const AABB& box) const {
    for (const auto& plane : planes) {
        // 找 AABB 上离平面法线方向最远的顶点（正顶点）
        glm::vec3 positive = box.min;
        if (plane.x >= 0)
            positive.x = box.max.x;
        if (plane.y >= 0)
            positive.y = box.max.y;
        if (plane.z >= 0)
            positive.z = box.max.z;

        // 若正顶点在平面外侧（距离 < 0），整个 AABB 不可见
        float dist = glm::dot(glm::vec3(plane), positive) + plane.w;
        if (dist < 0.0f)
            return false;
    }
    return true;
}

// ─── FrustumCuller ────────────────────────────────────────────────────────────

void FrustumCuller::cull(World& world, const Frustum& frustum, std::vector<EntityID>& outVisible) {
    outVisible.clear();
    lastTotal_ = 0;
    lastVisible_ = 0;

    world.view<MeshComponent, TransformComponent>([&](EntityID id, MeshComponent& mesh, TransformComponent& tc) {
        ++lastTotal_;

        AABB localBox{mesh.aabbMin, mesh.aabbMax};
        AABB worldBox = localBox.transform(tc.world);

        if (frustum.intersects(worldBox)) {
            outVisible.push_back(id);
            ++lastVisible_;
        }
    });
}

} // namespace engine
