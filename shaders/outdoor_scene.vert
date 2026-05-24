#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(binding = 0) uniform SceneUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 lightMVP;
    vec4 lightDir;
} ubo;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec4 shadowCoord;

void main()
{
    vec4 world  = ubo.model * vec4(inPosition, 1.0);
    fragPos     = world.xyz;
    fragNormal  = mat3(transpose(inverse(ubo.model))) * inNormal;
    fragColor   = inColor;
    shadowCoord = ubo.lightMVP * world;
    gl_Position = ubo.proj * ubo.view * world;
}
