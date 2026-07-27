#include <graphics/scene/camera.hpp>
#include <graphics/scene/camera_controller.hpp>

#include <glm/gtc/constants.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace vulkan_graphics;

namespace {

int failures = 0;

bool nearlyEqual(float left, float right, float epsilon = 1e-4f) {
    return std::abs(left - right) <= epsilon;
}

bool nearlyEqual(const glm::vec3& left, const glm::vec3& right, float epsilon = 1e-4f) {
    return glm::length(left - right) <= epsilon;
}

void expect(bool condition, const char* message) {
    if (condition)
        return;
    ++failures;
    std::cerr << "FAILED: " << message << '\n';
}

template <typename Function> void expectInvalidArgument(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    expect(false, message);
}

void testCameraMatrices() {
    Camera camera;
    camera.setPerspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 1000.0f);

    expect(nearlyEqual(camera.forward(), {0.0f, 0.0f, -1.0f}), "default camera should look along -Z");

    const glm::mat4 view = camera.viewMatrix();
    const glm::vec3 eyeInView = glm::vec3(view * glm::vec4(camera.position(), 1.0f));
    const glm::vec3 targetInView = glm::vec3(view * glm::vec4(camera.position() + camera.forward() * 5.0f, 1.0f));
    expect(nearlyEqual(eyeInView, {0.0f, 0.0f, 0.0f}), "view matrix should move the eye to the origin");
    expect(nearlyEqual(targetInView, {0.0f, 0.0f, -5.0f}), "view matrix should preserve forward distance");

    const glm::mat4 projection = camera.projectionMatrix();
    const glm::vec4 nearClip = projection * glm::vec4(0.0f, 0.0f, -0.1f, 1.0f);
    const glm::vec4 farClip = projection * glm::vec4(0.0f, 0.0f, -1000.0f, 1.0f);
    expect(nearlyEqual(nearClip.z / nearClip.w, 0.0f), "Vulkan near depth should be zero");
    expect(nearlyEqual(farClip.z / farClip.w, 1.0f), "Vulkan far depth should be one");

    const CameraMatrices matrices = camera.matrices();
    const glm::mat4 identity = matrices.viewProjection * matrices.inverseViewProjection;
    expect(nearlyEqual(identity[0][0], 1.0f) && nearlyEqual(identity[1][1], 1.0f) &&
               nearlyEqual(identity[2][2], 1.0f) && nearlyEqual(identity[3][3], 1.0f),
           "inverse view-projection matrix should invert view-projection");
}

void testFpsController() {
    Camera camera;
    FpsCameraController controller(camera);

    FpsCameraInput forward;
    forward.moveDirection.z = 1.0f;
    controller.update(forward, 1.0f);
    expect(nearlyEqual(camera.position(), {0.0f, 0.0f, 0.0f}), "FPS forward movement should use move speed");

    FpsCameraInput sprint;
    sprint.moveDirection.x = 1.0f;
    sprint.sprint = true;
    controller.update(sprint, 1.0f);
    expect(nearlyEqual(camera.position(), {15.0f, 0.0f, 0.0f}), "FPS sprint should multiply movement speed");

    controller.setAngles(glm::half_pi<float>(), 0.0f);
    expect(nearlyEqual(camera.forward(), {1.0f, 0.0f, 0.0f}), "FPS yaw should rotate the view around Y");
}

void testOrbitController() {
    Camera camera;
    OrbitCameraController controller(camera);
    expect(nearlyEqual(controller.target(), {0.0f, 0.0f, 0.0f}), "orbit controller should preserve the initial view");

    OrbitCameraInput zoom;
    zoom.zoomDelta = 1.0f;
    controller.update(zoom, 0.0f);
    expect(controller.distance() < 5.0f, "positive orbit zoom should reduce distance");
    expect(nearlyEqual(glm::length(camera.position() - controller.target()), controller.distance()),
           "orbit camera should stay on its configured radius");

    controller.setAngles(glm::half_pi<float>(), 0.0f);
    expect(camera.position().x > controller.target().x, "orbit yaw should move the eye around the target");
}

void testValidation() {
    Camera camera;
    expectInvalidArgument([&] { camera.setAspectRatio(0.0f); }, "zero aspect ratio should be rejected");
    expectInvalidArgument([&] { camera.setLookAt(glm::vec3{0.0f}, glm::vec3{0.0f}); },
                          "equal eye and target should be rejected");
    expectInvalidArgument([&] { camera.setPerspective(1.0f, 1.0f, 1.0f, 0.5f); },
                          "far plane behind near plane should be rejected");

    FpsCameraController controller(camera);
    expectInvalidArgument([&] { controller.update({}, -1.0f); }, "negative delta time should be rejected");
}

} // namespace

int main() {
    testCameraMatrices();
    testFpsController();
    testOrbitController();
    testValidation();

    if (failures != 0) {
        std::cerr << failures << " camera test(s) failed\n";
        return 1;
    }
    std::cout << "All VkKit camera tests passed\n";
    return 0;
}
