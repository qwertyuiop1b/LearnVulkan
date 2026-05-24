#version 450
// 简化 IBL：辐照度立方体贴图 + 预过滤环境 + BRDF LUT

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 cameraPos;
} ubo;

layout(set = 0, binding = 1) uniform samplerCube irradianceMap;
layout(set = 0, binding = 2) uniform samplerCube prefilteredMap;
layout(set = 0, binding = 3) uniform sampler2D brdfLUT;

const float PI = 3.14159265359;

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(ubo.cameraPos.xyz - fragWorldPos);
    vec3 R = reflect(-V, N);
    float NdotV = max(dot(N, V), 0.001);

    vec3 F0 = vec3(0.04);
    float metallic = 0.2;
    float roughness = 0.4;
    F0 = mix(F0, vec3(0.8, 0.6, 0.2), metallic);

    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse = irradiance;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilteredMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 envBRDF = texture(brdfLUT, vec2(NdotV, roughness)).rg;
    vec3 specular = prefilteredColor * (fresnelSchlickRoughness(NdotV, F0, roughness) * envBRDF.x + envBRDF.y);

    vec3 color = mix(diffuse, specular, metallic);
    outColor = vec4(color, 1.0);
}
