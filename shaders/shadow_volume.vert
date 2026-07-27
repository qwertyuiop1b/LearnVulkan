#version 450
layout(location = 0) in vec3 inPosition;
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 projection;
    vec4 lightPosition;
    vec4 cameraPosition;
} scene;
void main() {
    gl_Position = scene.projection * scene.view * vec4(inPosition, 1.0);
}
