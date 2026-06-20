#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(binding = 0) uniform SceneUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

void main()
{
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    fragPos    = worldPos.xyz;
    fragNormal = mat3(transpose(inverse(ubo.model))) * inNormal;
    fragColor  = inColor;
    gl_Position = ubo.proj * ubo.view * worldPos;
}
