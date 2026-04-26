#version 450
// 延迟光照通道片段着色器（第19章 修复版）
//
// 修复了以下问题：
//   1. 背景像素检测：albedo=0 的像素直接输出背景色，避免对无效 G-Buffer 数据做光照计算
//   2. albedo 双重乘法：原代码在循环内和循环外各乘一次，导致颜色偏暗

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inPosition;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput inNormal;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput inAlbedo;

struct Light {
    vec4 position;   // xyz = 位置，w = 半径
    vec4 color;      // rgb = 颜色，a = 强度
};

layout(set = 0, binding = 3) uniform LightUBO {
    Light  lights[4];
    vec4   cameraPos;
} ubo;

layout(location = 0) out vec4 outColor;

void main()
{
    // 从 G-Buffer 读取（subpassLoad 使用当前像素坐标，无需 UV）
    vec3 fragPos    = subpassLoad(inPosition).rgb;
    vec3 rawNormal  = subpassLoad(inNormal).rgb;
    vec3 albedo     = subpassLoad(inAlbedo).rgb;

    // ── 背景像素检测 ──────────────────────────────────────────────────────────
    // 未被几何体覆盖的像素：albedo G-Buffer = (0,0,0)
    // 若不跳过，fragPos=(0,0,0) 会当作世界原点被所有光源照亮，产生杂散亮点
    if (dot(albedo, albedo) < 0.001) {
        outColor = vec4(0.01, 0.01, 0.02, 1.0);   // 背景色
        return;
    }

    // 法线解码：[0,1] → [-1,1]（geometry pass 编码时用 normal*0.5+0.5）
    vec3 N = normalize(rawNormal * 2.0 - 1.0);
    vec3 V = normalize(ubo.cameraPos.xyz - fragPos);

    // ── 光照累加 ──────────────────────────────────────────────────────────────
    vec3 ambient  = vec3(0.06) * albedo;          // 环境光（含 albedo）
    vec3 diffuse  = vec3(0.0);
    vec3 specular = vec3(0.0);

    for (int i = 0; i < 4; ++i) {
        vec3  toLight = ubo.lights[i].position.xyz - fragPos;
        float dist    = length(toLight);
        float radius  = ubo.lights[i].position.w;
        if (dist > radius) continue;

        vec3  L    = normalize(toLight);
        float diff = max(dot(N, L), 0.0);

        // Blinn-Phong 高光
        vec3  H    = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), 32.0);

        // 平滑衰减（超出半径 50% 开始衰减）
        float atten = 1.0 - smoothstep(radius * 0.5, radius, dist);
        float intensity = ubo.lights[i].color.a * atten;

        diffuse  += diff * ubo.lights[i].color.rgb * intensity;
        specular += spec * ubo.lights[i].color.rgb * intensity * 0.3;
    }

    // ── 最终颜色（albedo 只乘一次）────────────────────────────────────────────
    vec3 result = ambient + diffuse * albedo + specular;
    outColor = vec4(result, 1.0);
}
