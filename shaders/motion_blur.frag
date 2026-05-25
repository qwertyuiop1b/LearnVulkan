#version 450
// 第72章：运动模糊 — 按速度方向采样
layout(location=0) in vec2 inUV;
layout(location=0) out vec4 outColor;

layout(set=0,binding=0) uniform sampler2D sceneTex;
layout(set=0,binding=1) uniform sampler2D velocityTex;

layout(push_constant) uniform BlurPC {
    float strength;   // 模糊强度（1.0=速度原始长度）
    int   samples;    // 采样数（8-16）
    float pad[2];
} pc;

void main() {
    vec2 vel    = texture(velocityTex, inUV).rg * pc.strength;
    float speed = length(vel);
    if (speed < 0.001) { outColor = texture(sceneTex, inUV); return; }

    vec3  color = vec3(0.0);
    int   n     = min(pc.samples, 16);
    for (int i = 0; i < n; ++i) {
        float t   = (float(i) / float(n-1)) - 0.5;   // [-0.5, 0.5]
        vec2  uv  = clamp(inUV + vel * t, 0.001, 0.999);
        color += texture(sceneTex, uv).rgb;
    }
    outColor = vec4(color / float(n), 1.0);
}
