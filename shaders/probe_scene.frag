#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform SceneUBO {
    mat4 model; mat4 view; mat4 proj;
    vec4 cameraPos;
    float roughness;
    float metallic;
    float pad[2];
} ubo;

layout(set = 0, binding = 1) uniform samplerCube probeCubemap;

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.cameraPos.xyz - fragPos);
    vec3 R = reflect(-V, N);

    vec3 lightDir  = normalize(vec3(1, 2, 1));
    float NdotL    = max(dot(N, lightDir), 0.0);
    vec3  diffuse  = fragColor * (NdotL * 0.8 + 0.2);

    // 从反射探针采样环境光
    vec3 envColor  = texture(probeCubemap, R).rgb;
    // Fresnel 近似（金属度越高 → 反射越强）
    float fresnel  = pow(1.0 - max(dot(N, V), 0.0), 2.0);
    float reflMix  = mix(ubo.roughness * 0.1, 1.0, ubo.metallic) * (0.04 + fresnel * 0.96);

    vec3 finalColor = mix(diffuse, envColor, clamp(reflMix, 0.0, 1.0));
    outColor = vec4(finalColor, 1.0);
}
