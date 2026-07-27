#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 projection;
    vec4 lightPosition;
    vec4 cameraPosition;
} scene;
layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 color;
void main() {
    worldPosition = inPosition;
    worldNormal = inNormal;
    color = inColor;
    gl_Position = scene.projection * scene.view * vec4(inPosition, 1.0);
}
