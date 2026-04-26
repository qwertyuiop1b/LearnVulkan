#version 450
// Post-Processing: HDR 场景片段着色器
// 输出线性 HDR 颜色（允许 >1.0 的亮度，Bloom 阶段用到）

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(set = 0, binding = 0) uniform UBO {
    mat4 model; mat4 view; mat4 projection;
    vec4 lightDir; vec4 lightColor; float exposure;
} ubo;

layout(location = 0) out vec4 outHDRColor;     // HDR 颜色（>1.0 允许）
layout(location = 1) out vec4 outBrightColor;  // 亮度超阈值的部分（用于 Bloom）

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-ubo.lightDir.xyz);

    float diff  = max(dot(N, L), 0.0);
    // 模拟高亮区域（亮度 * exposure 后超过 1.0）
    float brightness = 3.0;  // 刻意设高，使 Bloom 区域可见
    vec3 color = fragColor * (0.1 + diff * brightness) * ubo.lightColor.rgb;

    outHDRColor = vec4(color, 1.0);

    // 提取亮度超过阈值的像素（用于后续 Bloom blur 通道）
    // 阈值 1.0：只有超过白色亮度的部分才泛光
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    if (luminance > 1.0)
        outBrightColor = vec4(color, 1.0);
    else
        outBrightColor = vec4(0.0, 0.0, 0.0, 1.0);
}
