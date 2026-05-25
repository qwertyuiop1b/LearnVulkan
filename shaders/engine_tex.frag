#version 450
/// @brief 引擎 Demo 纹理片元着色器：采样纹理 + 光照

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;

void main() {
    vec3  lightDir = normalize(vec3(1.0, 2.0, 1.5));
    vec3  n        = normalize(fragNormal);
    float diff     = max(dot(n, lightDir), 0.0);
    vec4  texColor = texture(texSampler, fragTexCoord);
    float ambient  = 0.25;
    outColor       = vec4(texColor.rgb * (ambient + diff * 0.75), texColor.a);
}
