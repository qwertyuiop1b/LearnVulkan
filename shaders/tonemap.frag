#version 450
// Post-Processing: Tone Mapping + Bloom 合成 + Gamma Correction
//
// 完整后处理管线：
//   HDR Scene Buffer  ──┐
//                       ├── 合成 ──→ Tone Mapping ──→ Gamma ──→ 屏幕
//   Bloom Buffer      ──┘

layout(set = 0, binding = 0) uniform sampler2D hdrBuffer;    // HDR 场景颜色
layout(set = 0, binding = 1) uniform sampler2D bloomBuffer;  // Bloom 泛光

layout(push_constant) uniform PC {
    float exposure;      // 曝光度（调整整体亮度）
    float bloomStrength; // Bloom 混合强度
    float gamma;         // Gamma 校正值（通常 2.2）
} pc;

layout(location = 0) out vec4 outColor;

// Filmic ACES Tone Mapping（行业标准，效果最好）
// 来源：Stephen Hill 的 ACES 近似
vec3 ACESFilm(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

// Reinhard Tone Mapping（简单版）
vec3 ReinhardTonemap(vec3 hdr) {
    return hdr / (hdr + vec3(1.0));
}

// Uncharted 2 Tone Mapping（游戏行业常用）
vec3 Uncharted2Tonemap(vec3 x)
{
    float A=0.15,B=0.50,C=0.10,D=0.20,E=0.02,F=0.30;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

void main()
{
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(hdrBuffer, 0));

    // 1. 读取 HDR 颜色
    vec3 hdrColor = texture(hdrBuffer, uv).rgb;

    // 2. 叠加 Bloom
    vec3 bloomColor = texture(bloomBuffer, uv).rgb;
    hdrColor += bloomColor * pc.bloomStrength;

    // 3. 曝光调整（HDR 空间的亮度缩放）
    hdrColor *= pc.exposure;

    // 4. Tone Mapping：HDR [0,∞) → SDR [0,1]
    vec3 ldrColor = ACESFilm(hdrColor);

    // 5. Gamma 校正：线性空间 → 显示器 sRGB 空间
    // 显示器假设输入是 gamma=2.2 编码的，所以需要做 pow(color, 1/2.2)
    vec3 finalColor = pow(ldrColor, vec3(1.0 / pc.gamma));

    outColor = vec4(finalColor, 1.0);
}
