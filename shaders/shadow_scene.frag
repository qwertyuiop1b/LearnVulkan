#version 450
// 阴影场景渲染片段着色器（第18章）
// 通过 PCF（Percentage Closer Filtering）采样阴影贴图，产生柔和阴影

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec4 fragPosLightSpace;

layout(set = 0, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 lightSpaceMatrix;
    vec4 lightDir;
} ubo;

// 阴影贴图（深度纹理）
layout(set = 0, binding = 1) uniform sampler2D shadowMap;

layout(location = 0) out vec4 outColor;

// PCF 软阴影：对多个相邻像素采样取平均，模拟半影效果
float shadowCalc(vec4 lightSpacePos)
{
    // 透视除法：裁剪空间 → NDC
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;

    // Vulkan NDC 深度范围 [0,1]（OpenGL 是 [-1,1]）
    projCoords = projCoords * 0.5 + 0.5;

    // 超出阴影贴图范围的区域视为无阴影
    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z > 1.0) return 0.0;

    float currentDepth = projCoords.z;
    // bias：防止 shadow acne（自遮挡伪影）
    float bias = max(0.005 * (1.0 - dot(normalize(fragNormal),
                    -normalize(ubo.lightDir.xyz))), 0.001);

    // PCF: 3×3 区域采样，取平均值（9 次采样）
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap,
                projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += (currentDepth - bias > pcfDepth) ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}

void main()
{
    vec3 normal    = normalize(fragNormal);
    vec3 lightDir  = normalize(-ubo.lightDir.xyz);
    float diffuse  = max(dot(normal, lightDir), 0.0);
    float ambient  = 0.15;
    float shadow   = shadowCalc(fragPosLightSpace);

    // 阴影区域只有 ambient 光照
    float lighting = ambient + (1.0 - shadow) * diffuse;
    outColor = vec4(fragColor * lighting, 1.0);
}
