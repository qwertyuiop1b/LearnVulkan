#version 450
// 级联阴影场景片段着色器（第45章）
// 根据视图深度选择级联层，PCF 采样 shadow map 数组

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in float fragViewDepth;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 lightView;
    vec4 cascadeSplits;
    mat4 lightSpaceMatrix[3];
    vec4 lightDir;
} ubo;

layout(set = 0, binding = 1) uniform sampler2DArray shadowMap;

layout(location = 0) out vec4 outColor;

int selectCascade(float viewDepth)
{
    if (viewDepth < ubo.cascadeSplits.x) return 0;
    if (viewDepth < ubo.cascadeSplits.y) return 1;
    return 2;
}

float shadowCalc(vec3 normal, int cascade)
{
    mat4 lightSpace = ubo.lightSpaceMatrix[cascade];
    vec4 lightClip  = lightSpace * vec4(fragPos, 1.0);
    vec3 projCoords = lightClip.xyz / lightClip.w;
    projCoords = projCoords * 0.5 + 0.5;
    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z > 1.0)
        return 0.0;

    float bias = max(0.002 * (1.0 - dot(normalize(normal),
                    normalize(-ubo.lightDir.xyz))), 0.0005);
    float shadow = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0).xy);
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float depth = texture(shadowMap,
                vec3(projCoords.xy + vec2(x, y) * texel, float(cascade))).r;
            shadow += (projCoords.z - bias > depth) ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-ubo.lightDir.xyz);
    int cascade = selectCascade(fragViewDepth);
    float shadow = shadowCalc(N, cascade);
    float diff   = max(dot(N, L), 0.0);
    float lit    = 0.12 + (1.0 - shadow) * diff;
    // 级联边界可视化：不同级联略有色差（便于调试）
    vec3 tint = cascade == 0 ? vec3(1.0) :
                cascade == 1 ? vec3(0.98, 1.0, 0.98) : vec3(0.96, 0.98, 1.0);
    outColor = vec4(fragColor * lit * tint, 1.0);
}
