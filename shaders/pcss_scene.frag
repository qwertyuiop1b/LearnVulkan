#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec4 shadowCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform SceneUBO {
    mat4 model; mat4 view; mat4 proj;
    mat4 lightMVP;
    vec4 lightDir;
    vec4 lightColor;
    vec4 cameraPos;
    float lightSize;
    float shadowBias;
    float pad[2];
} ubo;

layout(set = 0, binding = 1) uniform sampler2D shadowMap;

// Poisson 采样圆盘（减少条带感）
const vec2 POISSON[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845), vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554), vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367), vec2( 0.14383161, -0.14100790)
);

// Step 1: Blocker Search — 找到在 searchRadius 内所有遮挡者的平均深度
float findBlockerDistance(vec2 uv, float currentDepth, float searchRadius)
{
    float blockerDepthSum = 0.0;
    int   blockerCount    = 0;
    for (int i = 0; i < 16; ++i) {
        vec2  sampleUV = uv + POISSON[i] * searchRadius;
        float shadowDepth = texture(shadowMap, sampleUV).r;
        if (shadowDepth < currentDepth - ubo.shadowBias) {
            blockerDepthSum += shadowDepth;
            ++blockerCount;
        }
    }
    if (blockerCount == 0) return -1.0;
    return blockerDepthSum / float(blockerCount);
}

// Step 2+3: PCF with PCSS-derived filter radius
float pcss(vec4 sc)
{
    // 透视除法到 [0,1]
    vec3 proj  = sc.xyz / sc.w;
    proj.xy    = proj.xy * 0.5 + 0.5;
    float depth = proj.z;

    if (depth > 1.0) return 1.0;

    // Blocker search
    float searchRadius = ubo.lightSize / depth * 0.01;
    float avgBlocker   = findBlockerDistance(proj.xy, depth, searchRadius);
    if (avgBlocker < 0.0) return 1.0;   // 没有遮挡者

    // Penumbra 半径估算
    float penumbra = (depth - avgBlocker) / avgBlocker * ubo.lightSize;
    float filterR  = clamp(penumbra * 0.05, 0.001, 0.02);

    // PCF 滤波
    float shadow = 0.0;
    for (int i = 0; i < 16; ++i) {
        vec2  sampleUV    = proj.xy + POISSON[i] * filterR;
        float shadowDepth = texture(shadowMap, sampleUV).r;
        shadow += (shadowDepth >= depth - ubo.shadowBias) ? 1.0 : 0.0;
    }
    return shadow / 16.0;
}

void main()
{
    vec3 normal   = normalize(fragNormal);
    vec3 lightDir = normalize(-ubo.lightDir.xyz);
    float NdotL   = max(dot(normal, lightDir), 0.0);

    float shadow  = pcss(shadowCoord);
    float ambient = 0.15;
    vec3  color   = fragColor * (ambient + NdotL * shadow * ubo.lightColor.rgb);

    outColor = vec4(color, 1.0);
}
