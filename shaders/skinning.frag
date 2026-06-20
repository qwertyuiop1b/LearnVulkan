#version 450
// 蒙皮网格片元：材质基色 + 简单光照

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;

layout(set = 0, binding = 2) uniform MaterialUBO {
    vec4 baseColor;
} material;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 n = normalize(fragNormal);
    vec3 lightDir = normalize(vec3(0.25, 1.0, 0.15));
    float diff = max(dot(n, lightDir), 0.08);
    outColor = vec4(material.baseColor.rgb * diff, material.baseColor.a);
}
