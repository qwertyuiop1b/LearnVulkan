#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

// 双输出：HDR 颜色 + 高亮区域（Bloom 提取）
layout(location = 0) out vec4 outHdr;
layout(location = 1) out vec4 outBright;

void main()
{
    vec3 lightDir = normalize(vec3(1.0, 2.0, 1.5));
    vec3 normal   = normalize(fragNormal);
    float diff    = max(dot(normal, lightDir), 0.0);
    vec3 ambient  = fragColor * 0.15;
    vec3 diffuse  = fragColor * diff;
    vec3 color    = ambient + diffuse;

    outHdr = vec4(color, 1.0);

    // 提取亮度超过阈值的区域作为 Bloom 源
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    outBright = brightness > 1.0 ? vec4(color, 1.0) : vec4(0.0);
}
