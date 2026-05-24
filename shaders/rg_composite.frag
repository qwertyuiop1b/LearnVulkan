#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrTex;
layout(set = 0, binding = 1) uniform sampler2D bloomTex;

layout(push_constant) uniform CompositePC {
    float bloomStrength;
    float exposure;
    int   debugView;   // 0=完整合成, 1=仅HDR, 2=仅Bloom, 3=仅亮区
} pc;

// ACES Filmic Tone Mapping（行业标准）
vec3 acesToneMap(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
    vec3 hdr   = texture(hdrTex,   inUV).rgb;
    vec3 bloom = texture(bloomTex, inUV).rgb;

    vec3 color;
    if (pc.debugView == 1) {
        // 仅 HDR（未 Tone Map）
        color = hdr;
    } else if (pc.debugView == 2) {
        // 仅 Bloom 层
        color = bloom;
    } else if (pc.debugView == 3) {
        // 亮区提取结果（debug）
        float bright = dot(hdr, vec3(0.2126, 0.7152, 0.0722));
        color = bright > 1.0 ? hdr : vec3(0.0);
    } else {
        // 完整合成：HDR + Bloom → Tone Map → Gamma
        color = hdr * pc.exposure + bloom * pc.bloomStrength;
        color = acesToneMap(color);
        color = pow(color, vec3(1.0 / 2.2));   // Gamma correction
    }

    outColor = vec4(color, 1.0);
}
