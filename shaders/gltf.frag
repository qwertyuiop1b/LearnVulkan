#version 450
// glTF 基础渲染：baseColorFactor + 可选漫反射贴图

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(set = 0, binding = 1) uniform MaterialUBO {
    vec4 baseColorFactor;
    vec3 lightDir;
    float _pad;
} material;

layout(set = 0, binding = 2) uniform sampler2D baseColorMap;

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-material.lightDir);
    float diff = max(dot(N, L), 0.15);
    vec3 albedo = texture(baseColorMap, fragTexCoord).rgb * material.baseColorFactor.rgb;
    outColor = vec4(albedo * diff, material.baseColorFactor.a);
}
