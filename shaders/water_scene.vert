#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(binding = 0) uniform SceneUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// 裁剪平面（通过 varying 传给 frag 做软裁剪，避免 gl_ClipDistance 扩展要求）
layout(push_constant) uniform ClipPC {
    vec4 clipPlane;   // 用于反射/折射 Pass 的裁剪平面
} pc;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out float clipDist;    // 传到 frag 做裁剪

void main()
{
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    fragPos    = worldPos.xyz;
    fragNormal = mat3(transpose(inverse(ubo.model))) * inNormal;
    fragColor  = inColor;
    gl_Position = ubo.proj * ubo.view * worldPos;
    // 传裁剪距离给 frag（不用 gl_ClipDistance，保证 MoltenVK 兼容性）
    clipDist   = dot(worldPos, pc.clipPlane);
}
