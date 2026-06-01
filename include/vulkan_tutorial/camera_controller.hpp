#pragma once

/**
 * @file camera_controller.hpp
 * @brief 轨道相机：鼠标旋转/平移/缩放 + WASD 平移目标点
 */

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace vulkan_tutorial {

class OrbitCamera {
  public:
    void setTarget(const glm::vec3& target) {
        target_ = target;
    }
    void setDistance(float distance) {
        distance_ = std::max(0.5f, distance);
    }
    void setAngles(float yawDegrees, float pitchDegrees) {
        yaw_ = yawDegrees;
        pitch_ = std::clamp(pitchDegrees, -89.0f, 89.0f);
    }
    void reset() {
        target_ = glm::vec3(0.0f);
        distance_ = 5.0f;
        yaw_ = 45.0f;
        pitch_ = 25.0f;
    }
    glm::vec3 target() const {
        return target_;
    }
    float distance() const {
        return distance_;
    }
    float yaw() const {
        return yaw_;
    }
    float pitch() const {
        return pitch_;
    }
    glm::vec3 eyePosition() const {
        const float yawRad = glm::radians(yaw_);
        const float pitchRad = glm::radians(pitch_);
        const glm::vec3 offset{distance_ * std::cos(pitchRad) * std::sin(yawRad),
                               distance_ * std::sin(pitchRad),
                               distance_ * std::cos(pitchRad) * std::cos(yawRad)};
        return target_ + offset;
    }
    glm::mat4 viewMatrix() const {
        return glm::lookAt(eyePosition(), target_, glm::vec3(0.0f, 1.0f, 0.0f));
    }
    glm::mat4
    projectionMatrix(float aspect, float fovDegrees = 45.0f, float nearPlane = 0.1f, float farPlane = 100.0f) const {
        glm::mat4 proj = glm::perspective(glm::radians(fovDegrees), aspect, nearPlane, farPlane);
        proj[1][1] *= -1.0f;
        return proj;
    }
    glm::mat4 viewMatrixWithoutTranslation() const {
        glm::mat4 view = viewMatrix();
        view[3] = glm::vec4(0.0f, 0.0f, 0.0f, view[3].w);
        return view;
    }
    void processKeyboard(GLFWwindow* window, float deltaSeconds) {
        if (!window)
            return;
        const float moveSpeed = 4.0f * deltaSeconds;
        const glm::vec3 forward =
            glm::normalize(glm::vec3(std::cos(glm::radians(yaw_)), 0.0f, std::sin(glm::radians(yaw_))));
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            target_ += forward * moveSpeed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            target_ -= forward * moveSpeed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            target_ -= right * moveSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            target_ += right * moveSpeed;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            target_.y -= moveSpeed;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            target_.y += moveSpeed;
    }
    void onMouseButton(int button, int action, double cursorX, double cursorY) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            rotating_ = action == GLFW_PRESS;
            if (rotating_) {
                lastCursorX_ = cursorX;
                lastCursorY_ = cursorY;
            }
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            panning_ = action == GLFW_PRESS;
            if (panning_) {
                lastCursorX_ = cursorX;
                lastCursorY_ = cursorY;
            }
        }
    }
    void onCursorMove(double cursorX, double cursorY) {
        const double dx = cursorX - lastCursorX_;
        const double dy = cursorY - lastCursorY_;
        lastCursorX_ = cursorX;
        lastCursorY_ = cursorY;
        if (rotating_) {
            yaw_ += static_cast<float>(dx) * 0.25f;
            pitch_ = std::clamp(pitch_ + static_cast<float>(dy) * 0.25f, -89.0f, 89.0f);
        }
        if (panning_) {
            const float panSpeed = distance_ * 0.002f;
            const glm::vec3 forward =
                glm::normalize(glm::vec3(std::cos(glm::radians(yaw_)), 0.0f, std::sin(glm::radians(yaw_))));
            const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
            target_ -= right * static_cast<float>(dx) * panSpeed;
            target_ += glm::vec3(0.0f, 1.0f, 0.0f) * static_cast<float>(dy) * panSpeed;
        }
    }
    void onScroll(double yOffset) {
        distance_ = std::max(0.5f, distance_ - static_cast<float>(yOffset) * 0.35f);
    }
    void onKey(int key, int action) {
        if (key == GLFW_KEY_R && action == GLFW_PRESS)
            reset();
    }

  private:
    glm::vec3 target_{0.0f};
    float distance_ = 5.0f;
    float yaw_ = 45.0f;
    float pitch_ = 25.0f;
    bool rotating_ = false;
    bool panning_ = false;
    double lastCursorX_ = 0.0;
    double lastCursorY_ = 0.0;
};

} // namespace vulkan_tutorial
