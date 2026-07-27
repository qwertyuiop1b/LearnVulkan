#include <graphics/scene/camera.hpp>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <stdexcept>

namespace vulkan_graphics {
namespace {

bool isFinite(const glm::vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool isFinite(const glm::quat& value) {
    return std::isfinite(value.w) && std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

void validateProjection(const PerspectiveProjection& projection) {
    constexpr float pi = 3.14159265358979323846f;
    if (!std::isfinite(projection.verticalFovRadians) || projection.verticalFovRadians <= 0.0f ||
        projection.verticalFovRadians >= pi)
        throw std::invalid_argument("Camera vertical FOV must be between 0 and pi radians");
    if (!std::isfinite(projection.aspectRatio) || projection.aspectRatio <= 0.0f)
        throw std::invalid_argument("Camera aspect ratio must be greater than zero");
    if (!std::isfinite(projection.nearPlane) || projection.nearPlane <= 0.0f)
        throw std::invalid_argument("Camera near plane must be greater than zero");
    if (!std::isfinite(projection.farPlane) || projection.farPlane <= projection.nearPlane)
        throw std::invalid_argument("Camera far plane must be greater than its near plane");
}

} // namespace

Camera::Camera(const PerspectiveProjection& projection) {
    setPerspective(projection);
}

void Camera::setPosition(const glm::vec3& position) {
    if (!isFinite(position))
        throw std::invalid_argument("Camera position must contain finite values");
    position_ = position;
}

void Camera::setOrientation(const glm::quat& orientation) {
    const float squaredLength = glm::dot(orientation, orientation);
    if (!isFinite(orientation) || !std::isfinite(squaredLength) || squaredLength <= 1e-12f)
        throw std::invalid_argument("Camera orientation must be a finite, non-zero quaternion");
    orientation_ = glm::normalize(orientation);
}

void Camera::setLookAt(const glm::vec3& eye, const glm::vec3& target, const glm::vec3& up) {
    if (!isFinite(eye) || !isFinite(target) || !isFinite(up))
        throw std::invalid_argument("Camera look-at vectors must contain finite values");

    const glm::vec3 offset = target - eye;
    const float offsetLength = glm::length(offset);
    const float upLength = glm::length(up);
    if (offsetLength <= 1e-6f)
        throw std::invalid_argument("Camera eye and target must be different");
    if (upLength <= 1e-6f)
        throw std::invalid_argument("Camera up vector must not be zero");

    const glm::vec3 direction = offset / offsetLength;
    const glm::vec3 normalizedUp = up / upLength;
    if (std::abs(glm::dot(direction, normalizedUp)) >= 0.9999f)
        throw std::invalid_argument("Camera direction and up vector must not be parallel");

    position_ = eye;
    orientation_ = glm::normalize(glm::quatLookAtRH(direction, normalizedUp));
}

void Camera::setPerspective(const PerspectiveProjection& projection) {
    validateProjection(projection);
    projection_ = projection;
}

void Camera::setPerspective(float verticalFovRadians, float aspectRatio, float nearPlane, float farPlane) {
    setPerspective({verticalFovRadians, aspectRatio, nearPlane, farPlane});
}

void Camera::setAspectRatio(float aspectRatio) {
    PerspectiveProjection updated = projection_;
    updated.aspectRatio = aspectRatio;
    setPerspective(updated);
}

const glm::vec3& Camera::position() const noexcept {
    return position_;
}

const glm::quat& Camera::orientation() const noexcept {
    return orientation_;
}

const PerspectiveProjection& Camera::perspective() const noexcept {
    return projection_;
}

glm::vec3 Camera::forward() const noexcept {
    return orientation_ * glm::vec3(0.0f, 0.0f, -1.0f);
}

glm::vec3 Camera::right() const noexcept {
    return orientation_ * glm::vec3(1.0f, 0.0f, 0.0f);
}

glm::vec3 Camera::up() const noexcept {
    return orientation_ * glm::vec3(0.0f, 1.0f, 0.0f);
}

glm::mat4 Camera::viewMatrix() const {
    const glm::mat4 inverseRotation = glm::mat4_cast(glm::conjugate(orientation_));
    return inverseRotation * glm::translate(glm::mat4(1.0f), -position_);
}

glm::mat4 Camera::projectionMatrix() const {
    glm::mat4 projection = glm::perspectiveRH_ZO(projection_.verticalFovRadians, projection_.aspectRatio,
                                                  projection_.nearPlane, projection_.farPlane);
    // VkKit uses a positive viewport height, so Vulkan's framebuffer Y is flipped here.
    projection[1][1] *= -1.0f;
    return projection;
}

glm::mat4 Camera::viewProjectionMatrix() const {
    return projectionMatrix() * viewMatrix();
}

CameraMatrices Camera::matrices() const {
    CameraMatrices result;
    result.view = viewMatrix();
    result.projection = projectionMatrix();
    result.viewProjection = result.projection * result.view;
    result.inverseViewProjection = glm::inverse(result.viewProjection);
    return result;
}

} // namespace vulkan_graphics
