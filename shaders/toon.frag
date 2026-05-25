#version 450
/**
 * 第84章：卡通着色器（Cel / Toon Shading）
 *
 * 技术要点：
 *  1. 量化漫反射（步进函数）— 硬色阶而非平滑渐变
 *  2. 硬高光（Blinn-Phong + 阈值二值化）
 *  3. 边缘光（Rim Light）— 相机角度接近90°时亮
 *  4. 多色调模式（2色/3色/4色/无限色）
 */

layout(location=0) in vec3 fragPos;
layout(location=1) in vec3 fragNormal;
layout(location=2) in vec3 fragColor;

layout(location=0) out vec4 outColor;

layout(binding=0) uniform ToonUBO {
    mat4 model, view, proj;
    vec4 lightDir;
    vec4 lightColor;
    vec4 cameraPos;
    float specularSize;
    float rimPower;
    float pad[2];
} ubo;

layout(push_constant) uniform ToonPC {
    int   shadingBands;   // 色阶数：2/3/4/0=平滑
    float shadowThreshold;// 阴影阈值（0.3-0.5）
    float shadowSmooth;   // 阴影边缘软化（0=硬, 0.1=软）
    int   enableRim;
    int   enableSpecular;
    int   enableOutlineColor;  // 在轮廓线章节中使用
    float outlineColorDark;    // 轮廓区域颜色暗化
    float pad;
} pc;

// 量化函数：将连续值转为离散色阶
float quantize(float val, int bands)
{
    if (bands <= 0) return val;   // 0 = 平滑（Phong）
    return floor(val * float(bands)) / float(bands);
}

void main()
{
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(ubo.lightDir.xyz);
    vec3 V = normalize(ubo.cameraPos.xyz - fragPos);
    vec3 H = normalize(L + V);

    // ── 1. 量化漫反射 ──────────────────────────────────────────────
    float NdotL = dot(N, L);

    float diff;
    if (pc.shadingBands <= 0) {
        // 平滑模式（普通 Phong）
        diff = max(NdotL, 0.0);
    } else {
        // 色阶量化
        // 先用 smoothstep 软化阴影边缘
        float smoothed = smoothstep(pc.shadowThreshold - pc.shadowSmooth,
                                    pc.shadowThreshold + pc.shadowSmooth,
                                    NdotL);
        diff = quantize(smoothed, pc.shadingBands);
    }

    // 环境光（避免全黑）
    float ambient = 0.2;

    // ── 2. 硬高光 ──────────────────────────────────────────────────
    float spec = 0.0;
    if (pc.enableSpecular != 0) {
        float NdotH  = max(dot(N, H), 0.0);
        float specRaw= pow(NdotH, 32.0);
        // 二值化：超过阈值 → 白色高光，否则 0
        float threshold = 1.0 - ubo.specularSize;
        spec = smoothstep(threshold - 0.01, threshold + 0.01, specRaw);
    }

    // ── 3. 边缘光（Rim Light）──────────────────────────────────────
    float rim = 0.0;
    if (pc.enableRim != 0) {
        float rimRaw = 1.0 - max(dot(N, V), 0.0);
        // 量化边缘光
        rim = smoothstep(0.6 - 0.05, 0.6 + 0.05, rimRaw);
        rim = pow(rim, ubo.rimPower);
    }

    // ── 合成 ───────────────────────────────────────────────────────
    vec3 baseColor = fragColor;

    // 应用量化漫反射：亮区用原色，暗区用深色版本
    vec3 shadedColor = baseColor * (ambient + diff * (1.0 - ambient));
    shadedColor     *= ubo.lightColor.rgb;

    // 叠加高光（白色）
    shadedColor += vec3(spec);

    // 叠加边缘光（蓝紫色调，卡通感）
    shadedColor += rim * vec3(0.3, 0.4, 0.8) * 0.5;

    outColor = vec4(shadedColor, 1.0);
}
