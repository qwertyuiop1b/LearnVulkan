#pragma once

#include <graphics/scene/camera.hpp>

#include <glm/glm.hpp>

namespace vulkan_graphics {

// Pixel deltas use window coordinates: +X is right and +Y is down.
struct OrbitCameraInput {
    glm::vec2 rotateDeltaPixels{0.0f};
    glm::vec2 panDeltaPixels{0.0f};
    float zoomDelta = 0.0f;
    // Local movement axes: +X right, +Y up, +Z forward.
    glm::vec3 moveDirection{0.0f};
};

struct OrbitCameraConfig {
    float rotationRadiansPerPixel = glm::radians(0.25f);
    float panUnitsPerPixelAtUnitDistance = 0.002f;
    float zoomPerStep = 0.12f;
    float moveSpeed = 4.0f;
    float minDistance = 0.1f;
    float maxDistance = 1000.0f;
    float maxPitchRadians = glm::radians(89.0f);
};

class OrbitCameraController final {
  public:
    explicit OrbitCameraController(Camera& camera, const OrbitCameraConfig& config = {});

    void update(const OrbitCameraInput& input, float deltaSeconds);
    void setTarget(const glm::vec3& target);
    void setDistance(float distance);
    void setAngles(float yawRadians, float pitchRadians);

    [[nodiscard]] const glm::vec3& target() const noexcept;
    [[nodiscard]] float distance() const noexcept;
    [[nodiscard]] float yaw() const noexcept;
    [[nodiscard]] float pitch() const noexcept;
    [[nodiscard]] const OrbitCameraConfig& config() const noexcept;

  private:
    void syncCamera();

    Camera& camera_;
    OrbitCameraConfig config_;
    glm::vec3 target_{0.0f};
    float distance_ = 5.0f;
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
};

struct FpsCameraInput {
    glm::vec2 lookDeltaPixels{0.0f};
    // Local movement axes: +X right, +Y up, +Z forward.
    glm::vec3 moveDirection{0.0f};
    bool sprint = false;
};

struct FpsCameraConfig {
    float lookRadiansPerPixel = glm::radians(0.1f);
    float moveSpeed = 5.0f;
    float sprintMultiplier = 3.0f;
    float maxPitchRadians = glm::radians(89.0f);
};

class FpsCameraController final {
  public:
    explicit FpsCameraController(Camera& camera, const FpsCameraConfig& config = {});

    void update(const FpsCameraInput& input, float deltaSeconds);
    void setPosition(const glm::vec3& position);
    void setAngles(float yawRadians, float pitchRadians);

    [[nodiscard]] float yaw() const noexcept;
    [[nodiscard]] float pitch() const noexcept;
    [[nodiscard]] const FpsCameraConfig& config() const noexcept;

  private:
    void syncCamera();

    Camera& camera_;
    FpsCameraConfig config_;
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
};

} // namespace vulkan_graphics
