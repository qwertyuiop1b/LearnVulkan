#version 450
// 第71章：景深 — 散景模糊（Poisson 圆盘采样）
layout(location=0) in vec2 inUV;
layout(location=0) out vec4 outColor;

layout(set=0,binding=0) uniform sampler2D cocTex;   // r=coc, gba=color

layout(push_constant) uniform BlurPC {
    vec2  texelSize;
    float blurScale;   // 散景大小倍率
    int   sampleCount; // 16 或 32
} pc;

// Poisson 圆盘采样点（16-tap）
const vec2 POISSON[16] = vec2[](
    vec2( 0.000, 0.000), vec2( 0.536, 0.000), vec2( 0.000, 0.536),
    vec2(-0.536, 0.000), vec2( 0.000,-0.536), vec2( 0.379, 0.379),
    vec2(-0.379, 0.379), vec2(-0.379,-0.379), vec2( 0.379,-0.379),
    vec2( 0.190, 0.507), vec2(-0.190, 0.507), vec2(-0.507, 0.190),
    vec2(-0.507,-0.190), vec2(-0.190,-0.507), vec2( 0.190,-0.507),
    vec2( 0.507,-0.190)
);

void main() {
    vec4  center = texture(cocTex, inUV);
    float coc    = center.r;
    float radius = abs(coc) * pc.blurScale;

    if (radius < 0.5) { outColor = vec4(center.gba, 1.0); return; }

    vec3  colorSum  = vec3(0.0);
    float weightSum = 0.0;

    for (int i = 0; i < 16; ++i) {
        vec2 offset   = POISSON[i] * radius * pc.texelSize;
        vec4 sample_  = texture(cocTex, inUV + offset);
        float sampleCoc = sample_.r;
        // 只让后景（正CoC）贡献前景模糊，反之亦然（简化版）
        float weight = (sign(coc) == sign(sampleCoc) || abs(sampleCoc) > abs(coc))
                       ? 1.0 : 0.2;
        colorSum  += sample_.gba * weight;
        weightSum += weight;
    }
    outColor = vec4(colorSum / weightSum, 1.0);
}
