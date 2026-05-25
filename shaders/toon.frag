#version 450
/**
 * 第84章（升级版）：卡通着色器
 *
 * 新增功能：
 *  1. 量化漫反射（色阶）
 *  2. 二值化高光
 *  3. 边缘光（Rim Light）
 *  4. 交叉线条阴影（Cross-Hatching）— 阴影区域显示倾斜线条，手绘感
 *  5. 调色板量化（Palette）— 将颜色映射到有限调色板
 *  6. 风格预设（Manga / Anime / Comic / Sketch）
 */

layout(location=0) in vec3 fragPos;
layout(location=1) in vec3 fragNormal;
layout(location=2) in vec3 fragColor;

layout(location=0) out vec4 outColor;
layout(location=1) out vec4 outNormal;   // 写入法线缓冲区（供边缘检测）

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
    int   shadingBands;
    float shadowThreshold;
    float shadowSmooth;
    int   enableRim;
    int   enableSpecular;
    int   stylePreset;     // 0=Anime, 1=Manga, 2=Comic, 3=Sketch
    float hatchDensity;    // 交叉线条密度
    int   enableHatching;
} pc;

// ── 工具函数 ──────────────────────────────────────────────────────────────

float quantize(float val, int bands) {
    if (bands <= 0) return val;
    return floor(val * float(bands)) / float(bands);
}

// 交叉线条（Cross-Hatching）— 用 UV 位置生成倾斜线条
// 返回 1 = 线条区域（深色），0 = 空白（正常）
float hatchPattern(vec3 pos, float density, int level)
{
    // level 0: 45° 线条
    // level 1: 135° 线条（与 level 0 交叉）
    // level 2: 垂直线条
    float scale = density * 15.0;
    float line0 = fract((pos.x + pos.z) * scale);
    float line1 = fract((pos.x - pos.z) * scale);
    float line2 = fract( pos.x           * scale);

    float width = 0.15;
    float h0 = step(1.0 - width, line0);
    float h1 = step(1.0 - width, line1);
    float h2 = step(1.0 - width, line2);

    if (level == 0) return h0;
    if (level == 1) return max(h0, h1);
    return max(max(h0, h1), h2);
}

// ── 风格预设参数 ──────────────────────────────────────────────────────────

struct StyleParams {
    int   bands;
    bool  monoChrome;    // 单色（漫画）
    float satBoost;      // 饱和度提升
    float contrastBoost;
    bool  useHatch;
    float hatchStart;    // 几阶以下开始 hatch（0-1）
};

StyleParams getStyle() {
    StyleParams s;
    s.bands         = pc.shadingBands;
    s.monoChrome    = false;
    s.satBoost      = 1.0;
    s.contrastBoost = 1.0;
    s.useHatch      = bool(pc.enableHatching);
    s.hatchStart    = 0.35;

    if (pc.stylePreset == 0) {  // Anime
        s.bands = 3; s.satBoost = 1.3;
    } else if (pc.stylePreset == 1) {  // Manga（B&W + 交叉线条）
        s.bands = 2; s.monoChrome = true; s.useHatch = true;
    } else if (pc.stylePreset == 2) {  // Comic
        s.bands = 4; s.satBoost = 1.5; s.contrastBoost = 1.2;
    } else if (pc.stylePreset == 3) {  // Sketch（更多线条，柔和色）
        s.bands = 3; s.useHatch = true; s.hatchStart = 0.5;
        s.satBoost = 0.7;
    }
    return s;
}

void main()
{
    StyleParams style = getStyle();

    vec3 N  = normalize(fragNormal);
    vec3 L  = normalize(ubo.lightDir.xyz);
    vec3 V  = normalize(ubo.cameraPos.xyz - fragPos);
    vec3 H  = normalize(L + V);

    float NdotL = dot(N, L);

    // ── 1. 量化漫反射 ──────────────────────────────────────────────────────
    float diff;
    float smoothed = smoothstep(pc.shadowThreshold - pc.shadowSmooth,
                                pc.shadowThreshold + pc.shadowSmooth,
                                NdotL);
    diff = quantize(smoothed, style.bands);

    // ── 2. 高光 ─────────────────────────────────────────────────────────────
    float spec = 0.0;
    if (pc.enableSpecular != 0) {
        float NdotH  = max(dot(N, H), 0.0);
        float specRaw= pow(NdotH, 32.0);
        float threshold = 1.0 - ubo.specularSize;
        spec = smoothstep(threshold - 0.01, threshold + 0.01, specRaw);
    }

    // ── 3. 边缘光 ────────────────────────────────────────────────────────────
    float rim = 0.0;
    if (pc.enableRim != 0) {
        float rimRaw = 1.0 - max(dot(N, V), 0.0);
        rim = smoothstep(0.55, 0.65, rimRaw);
        rim = pow(rim, ubo.rimPower);
    }

    // ── 4. 交叉线条（Cross-Hatching）────────────────────────────────────────
    if (style.useHatch && diff < style.hatchStart) {
        int hatchLevel = 0;
        if (diff < style.hatchStart * 0.33) hatchLevel = 2;      // 三层线条（极暗）
        else if (diff < style.hatchStart * 0.66) hatchLevel = 1; // 两层线条（暗）
        // else hatchLevel = 0                                     // 一层线条（半暗）

        float hatch = hatchPattern(fragPos, pc.hatchDensity, hatchLevel);
        // 叠加：线条区域颜色变深
        diff = mix(diff, diff * 0.3, hatch * (1.0 - diff / style.hatchStart));
    }

    // ── 5. 颜色合成 ──────────────────────────────────────────────────────────
    float ambient = 0.15;
    vec3 baseColor = fragColor;

    // 饱和度调整
    if (style.satBoost != 1.0) {
        float lum = dot(baseColor, vec3(0.2126, 0.7152, 0.0722));
        baseColor = mix(vec3(lum), baseColor, style.satBoost);
    }

    vec3 shadedColor = baseColor * (ambient + diff * (1.0 - ambient));
    shadedColor     *= ubo.lightColor.rgb;

    // 对比度
    if (style.contrastBoost != 1.0) {
        shadedColor = (shadedColor - 0.5) * style.contrastBoost + 0.5;
        shadedColor = clamp(shadedColor, 0.0, 1.0);
    }

    // 高光（白色叠加）
    shadedColor += vec3(spec);

    // 边缘光
    shadedColor += rim * vec3(0.3, 0.4, 0.9) * 0.5;

    // 漫画单色模式
    if (style.monoChrome) {
        float lum = dot(shadedColor, vec3(0.2126, 0.7152, 0.0722));
        shadedColor = vec3(lum);
    }

    outColor  = vec4(shadedColor, 1.0);
    outNormal = vec4(N * 0.5 + 0.5, 1.0);   // 法线编码到 [0,1] → 供边缘检测
}
