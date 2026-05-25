#pragma once
/**
 * @file scene_ecs.hpp
 * @brief 第68章：场景图 & 简单 ECS（Entity-Component-System）
 *
 * 设计：轻量级 ECS，教学目标是清楚展示 ECS 的核心思想：
 *   - Entity   = 一个 ID（uint32_t），无数据
 *   - Component = 纯数据结构（无方法，或仅有 trivial getter）
 *   - System   = 操作 Component 的函数 / Functor
 *   - World    = 注册 / 查询 / 遍历 的中心对象
 *
 * 对比 OOP 场景图（Node 持有子节点列表）：
 *   - ECS 的 Component 以 SoA（数组结构）存储，Cache-Friendly
 *   - 遍历同类 Component 不需要虚函数和 dynamic_cast
 *
 * 附加功能：
 *   - AABB 包围盒 + 视锥裁剪（FrustumCuller）
 *   - 变换矩阵脏标记传播（TransformSystem）
 */

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace engine {

// ─── Entity ───────────────────────────────────────────────────────────────

using EntityID = uint32_t;
constexpr EntityID NULL_ENTITY = UINT32_MAX;

// ─── Components ───────────────────────────────────────────────────────────

/** 位置 / 旋转 / 缩放 */
struct TransformComponent {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f};   ///< Euler angles（度）
    glm::vec3 scale   {1.0f};
    EntityID  parent  = NULL_ENTITY;

    mutable glm::mat4 world{1.0f};   ///< 由 TransformSystem 填充
    mutable bool      dirty = true;

    [[nodiscard]] glm::mat4 localMatrix() const;
};

/** 渲染网格引用（仅存 ID/指针，不持有 GPU 资源）*/
struct MeshComponent {
    VkBuffer   vertexBuffer  = VK_NULL_HANDLE;
    VkBuffer   indexBuffer   = VK_NULL_HANDLE;
    uint32_t   indexCount    = 0;
    uint32_t   vertexCount   = 0;
    glm::vec3  aabbMin{-0.5f};
    glm::vec3  aabbMax{ 0.5f};
};

/** 材质引用 */
struct MaterialComponent {
    uint32_t materialId  = 0;     ///< 对应 MaterialLibrary 中的索引
    bool     castShadow  = true;
    bool     receiveShadow = true;
    float    opacity     = 1.0f;
};

/** 平行光 */
struct DirectionalLightComponent {
    glm::vec3 direction{-1,-2,-1};
    glm::vec3 color    {1, 1, 1};
    float     intensity = 1.0f;
};

/** 点光源 */
struct PointLightComponent {
    glm::vec3 color    {1, 1, 1};
    float     intensity = 1.0f;
    float     radius    = 10.0f;
};

/** 相机 */
struct CameraComponent {
    float fovDegrees = 45.0f;
    float nearPlane  = 0.1f;
    float farPlane   = 1000.0f;
    bool  isMain     = true;

    [[nodiscard]] glm::mat4 projMatrix(float aspect) const;
};

/** 任意字符串标签（调试用）*/
struct TagComponent {
    std::string name;
};

// ─── ComponentStorage ─────────────────────────────────────────────────────

/**
 * @brief 泛型组件存储（密集数组 + EntityID → 索引 映射）
 *
 * Dense array = Cache-Friendly 遍历
 * 支持：add / remove / get / contains / forEach
 */
template<typename T>
class ComponentStorage {
public:
    void add(EntityID id, T component)
    {
        if (indexOf_.count(id)) return;   // 已有则忽略
        indexOf_[id] = static_cast<uint32_t>(entities_.size());
        entities_.push_back(id);
        components_.push_back(std::move(component));
    }

    void remove(EntityID id)
    {
        auto it = indexOf_.find(id);
        if (it == indexOf_.end()) return;
        uint32_t idx  = it->second;
        uint32_t last = static_cast<uint32_t>(entities_.size()) - 1;
        if (idx != last) {
            std::swap(entities_[idx],   entities_[last]);
            std::swap(components_[idx], components_[last]);
            indexOf_[entities_[idx]] = idx;
        }
        entities_.pop_back();
        components_.pop_back();
        indexOf_.erase(it);
    }

    [[nodiscard]] T*       get(EntityID id)
    {
        auto it = indexOf_.find(id);
        return it != indexOf_.end() ? &components_[it->second] : nullptr;
    }
    [[nodiscard]] const T* get(EntityID id) const
    {
        auto it = indexOf_.find(id);
        return it != indexOf_.end() ? &components_[it->second] : nullptr;
    }
    [[nodiscard]] bool contains(EntityID id) const { return indexOf_.count(id) > 0; }
    [[nodiscard]] size_t size() const { return components_.size(); }

    void forEach(std::function<void(EntityID, T&)> fn)
    {
        for (size_t i = 0; i < entities_.size(); ++i)
            fn(entities_[i], components_[i]);
    }
    void forEach(std::function<void(EntityID, const T&)> fn) const
    {
        for (size_t i = 0; i < entities_.size(); ++i)
            fn(entities_[i], components_[i]);
    }

    [[nodiscard]] std::vector<T>&       data()       { return components_; }
    [[nodiscard]] const std::vector<T>& data() const { return components_; }

private:
    std::vector<EntityID>               entities_;
    std::vector<T>                      components_;
    std::unordered_map<EntityID, uint32_t> indexOf_;
};

// ─── World ────────────────────────────────────────────────────────────────

/**
 * @brief ECS 世界 —— 实体和组件的注册中心
 *
 * 使用示例：
 * @code
 *   World world;
 *   EntityID box = world.createEntity("MyBox");
 *   world.add<TransformComponent>(box, {{1,0,0}});
 *   world.add<MeshComponent>(box, {vb, ib, 36});
 *   world.add<MaterialComponent>(box, {matId});
 *
 *   // 遍历所有有 Mesh 和 Transform 的实体：
 *   world.view<MeshComponent, TransformComponent>(
 *       [](EntityID e, MeshComponent& m, TransformComponent& t) {
 *           // 渲染...
 *       });
 * @endcode
 */
class World {
public:
    ~World() { for (auto& fn : destroyers_) fn(); }

    [[nodiscard]] EntityID createEntity(const std::string& tag = "");
    void destroyEntity(EntityID id);
    [[nodiscard]] bool isAlive(EntityID id) const;
    [[nodiscard]] EntityID findByTag(const std::string& tag) const;

    template<typename T>
    void add(EntityID id, T component)
    {
        storage<T>().add(id, std::move(component));
    }

    template<typename T>
    void remove(EntityID id) { storage<T>().remove(id); }

    template<typename T>
    [[nodiscard]] T* get(EntityID id) { return storage<T>().get(id); }

    template<typename T>
    [[nodiscard]] const T* get(EntityID id) const { return storage<T>().get(id); }

    template<typename T>
    [[nodiscard]] bool has(EntityID id) const { return storage<T>().contains(id); }

    template<typename T>
    void forEach(std::function<void(EntityID, T&)> fn)
    {
        storage<T>().forEach(fn);
    }

    /// 遍历同时拥有两种 Component 的实体
    template<typename A, typename B>
    void view(std::function<void(EntityID, A&, B&)> fn)
    {
        storage<A>().forEach([&](EntityID id, A& a) {
            B* b = storage<B>().get(id);
            if (b) fn(id, a, *b);
        });
    }

    /// 遍历同时拥有三种 Component 的实体
    template<typename A, typename B, typename C>
    void view(std::function<void(EntityID, A&, B&, C&)> fn)
    {
        storage<A>().forEach([&](EntityID id, A& a) {
            B* b = storage<B>().get(id);
            C* c = storage<C>().get(id);
            if (b && c) fn(id, a, *b, *c);
        });
    }

    [[nodiscard]] size_t entityCount() const { return aliveCount_; }

    template<typename T>
    [[nodiscard]] ComponentStorage<T>& storage();

private:
    EntityID  nextId_    = 0;
    size_t    aliveCount_= 0;
    std::vector<EntityID> alive_;

    // 存储 ComponentStorage 的 void* 映射（类型擦除）
    std::unordered_map<size_t, void*> stores_;
    std::vector<std::function<void()>> destroyers_;

    template<typename T>
    [[nodiscard]] const ComponentStorage<T>& storage() const;
};

// ─── 内置 Systems ─────────────────────────────────────────────────────────

/// 递归更新脏 TransformComponent 的世界矩阵
void transformSystem(World& world);

// ─── AABB 与视锥裁剪 ──────────────────────────────────────────────────────

struct AABB {
    glm::vec3 min{-0.5f};
    glm::vec3 max{ 0.5f};

    [[nodiscard]] AABB transform(const glm::mat4& m) const;
    [[nodiscard]] glm::vec3 center()  const { return (min + max) * 0.5f; }
    [[nodiscard]] glm::vec3 extents() const { return (max - min) * 0.5f; }
};

struct Frustum {
    glm::vec4 planes[6];   ///< Ax + By + Cz + D = 0，法线朝内

    static Frustum fromViewProj(const glm::mat4& viewProj);
    [[nodiscard]] bool intersects(const AABB& box) const;
};

/**
 * @brief 视锥裁剪系统
 *
 * 遍历所有 (MeshComponent + TransformComponent)，
 * 测试其 AABB 是否与视锥相交，输出可见 EntityID 列表。
 */
class FrustumCuller {
public:
    void cull(World& world, const Frustum& frustum,
              std::vector<EntityID>& outVisible);

    [[nodiscard]] uint32_t lastTotalCount()   const { return lastTotal_; }
    [[nodiscard]] uint32_t lastVisibleCount() const { return lastVisible_; }
    [[nodiscard]] float    lastCullRatio()    const
    {
        return lastTotal_ > 0
            ? float(lastTotal_ - lastVisible_) / float(lastTotal_)
            : 0.0f;
    }

private:
    uint32_t lastTotal_   = 0;
    uint32_t lastVisible_ = 0;
};

// ─── World::storage<T> 特化辅助 ──────────────────────────────────────────
// (实现在 .hpp 内，模板不能分离定义)

template<typename T>
ComponentStorage<T>& World::storage()
{
    size_t key = typeid(T).hash_code();
    auto it = stores_.find(key);
    if (it == stores_.end()) {
        auto* s = new ComponentStorage<T>();
        stores_[key] = s;
        destroyers_.push_back([s]{ delete s; });
        return *s;
    }
    return *reinterpret_cast<ComponentStorage<T>*>(it->second);
}

template<typename T>
const ComponentStorage<T>& World::storage() const
{
    size_t key = typeid(T).hash_code();
    auto it = stores_.find(key);
    if (it == stores_.end())
        throw std::runtime_error("ComponentStorage not found for type");
    return *reinterpret_cast<const ComponentStorage<T>*>(it->second);
}

} // namespace engine
