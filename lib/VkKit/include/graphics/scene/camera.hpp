#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace vulkan_graphics {

struct PerspectiveProjection {
    float verticalFovRadians = glm::radians(45.0f);
    float aspectRatio = 1.0f;
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;
};

struct CameraMatrices {
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::mat4 viewProjection{1.0f};
    glm::mat4 inverseViewProjection{1.0f};
};

// Right-handed, Y-up camera. Its local forward axis is -Z.
class Camera final {
  public:
    Camera() = default;
    explicit Camera(const PerspectiveProjection& projection);

    void setPosition(const glm::vec3& position);
    void setOrientation(const glm::quat& orientation);
    void setLookAt(const glm::vec3& eye, const glm::vec3& target,
                   const glm::vec3& up = {0.0f, 1.0f, 0.0f});

    void setPerspective(const PerspectiveProjection& projection);
    void setPerspective(float verticalFovRadians, float aspectRatio, float nearPlane, float farPlane);
    void setAspectRatio(float aspectRatio);

    [[nodiscard]] const glm::vec3& position() const noexcept;
    [[nodiscard]] const glm::quat& orientation() const noexcept;
    [[nodiscard]] const PerspectiveProjection& perspective() const noexcept;

    [[nodiscard]] glm::vec3 forward() const noexcept;
    [[nodiscard]] glm::vec3 right() const noexcept;
    [[nodiscard]] glm::vec3 up() const noexcept;

    [[nodiscard]] glm::mat4 viewMatrix() const;
    [[nodiscard]] glm::mat4 projectionMatrix() const;
    [[nodiscard]] glm::mat4 viewProjectionMatrix() const;
    [[nodiscard]] CameraMatrices matrices() const;

  private:
    glm::vec3 position_{0.0f, 0.0f, 5.0f};
    glm::quat orientation_{1.0f, 0.0f, 0.0f, 0.0f};
    PerspectiveProjection projection_{};
};

} // namespace vulkan_graphics
