#version 450
// 加权混合 OIT 累积通道（MRT：accumulation + revealage）
// 参考 McGuire & Bavoil, "Weighted Blended Order-Independent Transparency"

layout(location = 0) in vec3 fragColor;
layout(location = 1) in float fragAlpha;

layout(location = 0) out vec4 outAccum;
layout(location = 1) out vec4 outReveal;

float computeWeight(float z, float alpha) {
    // 深度与透明度共同决定权重，近处/高 alpha 贡献更大
    return clamp(pow(min(1.0, alpha * 10.0) + 0.01, 3.0) * 1e8 * pow(1.0 - z, 3.0),
                 1e-2, 3e3);
}

void main() {
    float alpha = clamp(fragAlpha, 0.0, 1.0);
    if (alpha < 0.01)
        discard;
    float z = gl_FragCoord.z;
    float w = computeWeight(z, alpha);
    outAccum = vec4(fragColor * alpha, alpha) * w;
    outReveal = vec4(alpha);
}
