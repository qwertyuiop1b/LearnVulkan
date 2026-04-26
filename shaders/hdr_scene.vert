#version 450
// Post-Processing: 场景渲染（HDR 颜色空间）
// 渲染到 RGBA16F 帧缓冲，允许亮度超过 1.0

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(set = 0, binding = 0) uniform UBO {
    mat4 model; mat4 view; mat4 projection;
    vec4 lightDir;  vec4 lightColor;  float exposure;
} ubo;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

void main()
{
    vec4 worldPos   = ubo.model * vec4(inPosition, 1.0);
    gl_Position     = ubo.projection * ubo.view * worldPos;
    fragPos         = worldPos.xyz;
    fragNormal      = normalize(mat3(transpose(inverse(ubo.model))) * inNormal);
    fragColor       = inColor;
}
