#version 450
// 阴影场景渲染顶点着色器（第18章）
// 同时输出裁剪空间坐标和光源空间坐标（用于阴影采样）

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(set = 0, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 lightSpaceMatrix;  // 光源 MVP，用于计算阴影坐标
    vec4 lightDir;
} ubo;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec4 fragPosLightSpace;  // 光源空间坐标

void main()
{
    vec4 worldPos    = ubo.model * vec4(inPosition, 1.0);
    gl_Position      = ubo.projection * ubo.view * worldPos;
    fragPos          = worldPos.xyz;
    fragNormal       = mat3(transpose(inverse(ubo.model))) * inNormal;
    fragColor        = inColor;
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;
}
