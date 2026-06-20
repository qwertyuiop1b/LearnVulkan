#version 450
// 级联阴影场景顶点着色器（第45章）
// 输出视图空间深度，供片段着色器选择级联层

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 lightView;
    vec4 cascadeSplits;           // xyz = 3 个分割点, w = near
    mat4 lightSpaceMatrix[3];     // 每级联的光源空间矩阵
    vec4 lightDir;
} ubo;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out float fragViewDepth;

void main()
{
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    vec4 viewPos  = ubo.view * worldPos;
    gl_Position   = ubo.projection * viewPos;
    fragPos       = worldPos.xyz;
    fragNormal    = mat3(transpose(inverse(ubo.model))) * inNormal;
    fragColor     = inColor;
    fragViewDepth = -viewPos.z;
}
