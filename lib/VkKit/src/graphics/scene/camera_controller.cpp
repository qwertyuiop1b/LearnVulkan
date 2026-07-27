#include <graphics/scene/camera_controller.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vulkan_graphics {
namespace {

constexpr glm::vec3 worldUp{0.0f, 1.0f, 0.0f};

bool isFinite(const glm::vec2& value) {
    return std::isfinite(value.x) && std::isfinite(value.y);
}

bool isFinite(const glm::vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void validateDeltaSeconds(float deltaSeconds) {
    if (!std::isfinite(deltaSeconds) || deltaSeconds < 0.0f)
        throw std::invalid_argument("Camera controller delta time must be finite and non-negative");
}

void validateOrbitConfig(const OrbitCameraConfig& config) {
    if (!std::isfinite(config.rotationRadiansPerPixel) || config.rotationRadiansPerPixel < 0.0f ||
        !std::isfinite(config.panUnitsPerPixelAtUnitDistance) || config.panUnitsPerPixelAtUnitDistance < 0.0f ||
        !std::isfinite(config.zoomPerStep) || config.zoomPerStep < 0.0f || !std::isfinite(config.moveSpeed) ||
        config.moveSpeed < 0.0f || !std::isfinite(config.minDistance) || config.minDistance <= 0.0f ||
        !std::isfinite(config.maxDistance) || config.maxDistance < config.minDistance ||
        !std::isfinite(config.maxPitchRadians) || config.maxPitchRadians <= 0.0f ||
        config.maxPitchRadians >= glm::radians(90.0f))
        throw std::invalid_argument("Orbit camera configuration contains an invalid range or speed");
}

void validateFpsConfig(const FpsCameraConfig& config) {
    if (!std::isfinite(config.lookRadiansPerPixel) || config.lookRadiansPerPixel < 0.0f ||
        !std::isfinite(config.moveSpeed) || config.moveSpeed < 0.0f || !std::isfinite(config.sprintMultiplier) ||
        config.sprintMultiplier < 1.0f || !std::isfinite(config.maxPitchRadians) ||
        config.maxPitchRadians <= 0.0f || config.maxPitchRadians >= glm::radians(90.0f))
        throw std::invalid_argument("FPS camera configuration contains an invalid range or speed");
}

glm::vec3 normalizedInputDirection(const glm::vec3& direction) {
    if (!isFinite(direction))
        throw std::invalid_argument("Camera movement input must contain finite values");
    const float length = glm::length(direction);
    return length > 1.0f ? direction / length : direction;
}

} // namespace

OrbitCameraController::OrbitCameraController(Camera& camera, const OrbitCameraConfig& config)
    : camera_(camera), config_(config) {
    validateOrbitConfig(config_);
    distance_ = std::clamp(5.0f, config_.minDistance, config_.maxDistance);
    target_ = camera_.position() + camera_.forward() * distance_;

    const glm::vec3 offset = camera_.position() - target_;
    yaw_ = std::atan2(offset.x, offset.z);
    pitch_ = std::asin(std::clamp(offset.y / distance_, -1.0f, 1.0f));
}

void OrbitCameraController::update(const OrbitCameraInput& input, float deltaSeconds) {
    validateDeltaSeconds(deltaSeconds);
    if (!isFinite(input.rotateDeltaPixels) || !isFinite(input.panDeltaPixels) || !std::isfinite(input.zoomDelta))
        throw std::invalid_argument("Orbit camera input must contain finite values");

    yaw_ += input.rotateDeltaPixels.x * config_.rotationRadiansPerPixel;
    pitch_ = std::clamp(pitch_ - input.rotateDeltaPixels.y * config_.rotationRadiansPerPixel,
                        -config_.maxPitchRadians, config_.maxPitchRadians);

    const float panScale = distance_ * config_.panUnitsPerPixelAtUnitDistance;
    target_ -= camera_.right() * input.panDeltaPixels.x * panScale;
    target_ += camera_.up() * input.panDeltaPixels.y * panScale;

    distance_ *= std::exp(-input.zoomDelta * config_.zoomPerStep);
    distance_ = std::clamp(distance_, config_.minDistance, config_.maxDistance);

    const glm::vec3 movement = normalizedInputDirection(input.moveDirection);
    glm::vec3 flatForward{std::sin(yaw_), 0.0f, -std::cos(yaw_)};
    const glm::vec3 flatRight{std::cos(yaw_), 0.0f, std::sin(yaw_)};
    target_ += (flatRight * movement.x + worldUp * movement.y + flatForward * movement.z) *
               config_.moveSpeed * deltaSeconds;

    syncCamera();
}

void OrbitCameraController::setTarget(const glm::vec3& target) {
    if (!isFinite(target))
        throw std::invalid_argument("Orbit camera target must contain finite values");
    target_ = target;
    syncCamera();
}

void OrbitCameraController::setDistance(float distance) {
    if (!std::isfinite(distance))
        throw std::invalid_argument("Orbit camera distance must be finite");
    distance_ = std::clamp(distance, config_.minDistance, config_.maxDistance);
    syncCamera();
}

void OrbitCameraController::setAngles(float yawRadians, float pitchRadians) {
    if (!std::isfinite(yawRadians) || !std::isfinite(pitchRadians))
        throw std::invalid_argument("Orbit camera angles must be finite");
    yaw_ = yawRadians;
    pitch_ = std::clamp(pitchRadians, -config_.maxPitchRadians, config_.maxPitchRadians);
    syncCamera();
}

const glm::vec3& OrbitCameraController::target() const noexcept {
    return target_;
}

float OrbitCameraController::distance() const noexcept {
    return distance_;
}

float OrbitCameraController::yaw() const noexcept {
    return yaw_;
}

float OrbitCameraController::pitch() const noexcept {
    return pitch_;
}

const OrbitCameraConfig& OrbitCameraController::config() const noexcept {
    return config_;
}

void OrbitCameraController::syncCamera() {
    const float cosPitch = std::cos(pitch_);
    const glm::vec3 offset{distance_ * cosPitch * std::sin(yaw_), distance_ * std::sin(pitch_),
                           distance_ * cosPitch * std::cos(yaw_)};
    camera_.setLookAt(target_ + offset, target_, worldUp);
}

FpsCameraController::FpsCameraController(Camera& camera, const FpsCameraConfig& config)
    : camera_(camera), config_(config) {
    validateFpsConfig(config_);
    const glm::vec3 direction = glm::normalize(camera_.forward());
    yaw_ = std::atan2(direction.x, -direction.z);
    pitch_ = std::asin(std::clamp(direction.y, -1.0f, 1.0f));
}

void FpsCameraController::update(const FpsCameraInput& input, float deltaSeconds) {
    validateDeltaSeconds(deltaSeconds);
    if (!isFinite(input.lookDeltaPixels))
        throw std::invalid_argument("FPS camera look input must contain finite values");

    yaw_ += input.lookDeltaPixels.x * config_.lookRadiansPerPixel;
    pitch_ = std::clamp(pitch_ - input.lookDeltaPixels.y * config_.lookRadiansPerPixel,
                        -config_.maxPitchRadians, config_.maxPitchRadians);

    const glm::vec3 movement = normalizedInputDirection(input.moveDirection);
    const glm::vec3 flatForward{std::sin(yaw_), 0.0f, -std::cos(yaw_)};
    const glm::vec3 flatRight{std::cos(yaw_), 0.0f, std::sin(yaw_)};
    const float speed = config_.moveSpeed * (input.sprint ? config_.sprintMultiplier : 1.0f);
    const glm::vec3 displacement =
        (flatRight * movement.x + worldUp * movement.y + flatForward * movement.z) * speed * deltaSeconds;
    camera_.setPosition(camera_.position() + displacement);
    syncCamera();
}

void FpsCameraController::setPosition(const glm::vec3& position) {
    camera_.setPosition(position);
}

void FpsCameraController::setAngles(float yawRadians, float pitchRadians) {
    if (!std::isfinite(yawRadians) || !std::isfinite(pitchRadians))
        throw std::invalid_argument("FPS camera angles must be finite");
    yaw_ = yawRadians;
    pitch_ = std::clamp(pitchRadians, -config_.maxPitchRadians, config_.maxPitchRadians);
    syncCamera();
}

float FpsCameraController::yaw() const noexcept {
    return yaw_;
}

float FpsCameraController::pitch() const noexcept {
    return pitch_;
}

const FpsCameraConfig& FpsCameraController::config() const noexcept {
    return config_;
}

void FpsCameraController::syncCamera() {
    const float cosPitch = std::cos(pitch_);
    const glm::vec3 direction{cosPitch * std::sin(yaw_), std::sin(pitch_), -cosPitch * std::cos(yaw_)};
    camera_.setLookAt(camera_.position(), camera_.position() + direction, worldUp);
}

} // namespace vulkan_graphics
