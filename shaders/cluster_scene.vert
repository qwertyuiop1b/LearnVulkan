#version 450
// 第76章：聚簇前向着色 — 顶点着色器
layout(location=0) in vec3 inPosition;
layout(location=1) in vec3 inNormal;
layout(location=2) in vec3 inColor;

layout(binding=0) uniform SceneUBO {
    mat4 model, view, proj;
} ubo;

layout(location=0) out vec3 fragPos;
layout(location=1) out vec3 fragNormal;
layout(location=2) out vec3 fragColor;

void main() {
    vec4 world = ubo.model * vec4(inPosition,1.0);
    fragPos    = world.xyz;
    fragNormal = mat3(transpose(inverse(ubo.model))) * inNormal;
    fragColor  = inColor;
    gl_Position = ubo.proj * ubo.view * world;
}
