#version 450
// TAA 场景顶点：应用投影抖动

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragNormal;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 prevViewProj;
    vec2 jitter;
    vec2 resolution;
} ubo;

void main() {
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    vec4 clipPos = ubo.projection * ubo.view * worldPos;
    clipPos.xy += ubo.jitter * clipPos.w;
    gl_Position = clipPos;
    fragNormal = mat3(ubo.model) * inNormal;
}
