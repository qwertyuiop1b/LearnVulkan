#version 450
/// @brief 引擎 Demo 基础片元着色器：漫反射光照

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 lightDir = normalize(vec3(1.0, 2.0, 1.5));
    vec3 n        = normalize(fragNormal);
    float diff    = max(dot(n, lightDir), 0.0);
    vec3 ambient  = fragColor * 0.2;
    vec3 result   = ambient + fragColor * diff * 0.8;
    outColor      = vec4(result, 1.0);
}
