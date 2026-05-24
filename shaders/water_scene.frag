#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in float clipDist;

layout(location = 0) out vec4 outColor;

void main()
{
    // 软裁剪：裁剪距离为负则输出透明（RTT Pass 使用，Final Pass clipPlane=0 无影响）
    if (clipDist < 0.0) {
        outColor = vec4(0.0);
        return;
    }
    vec3 lightDir = normalize(vec3(1.0, 3.0, 1.5));
    vec3 normal   = normalize(fragNormal);
    float diff    = max(dot(normal, lightDir), 0.0) * 0.8 + 0.2;
    outColor = vec4(fragColor * diff, 1.0);
}
