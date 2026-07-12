#version 450

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0, std430) readonly buffer Samples { vec4 values[]; };
layout(push_constant) uniform Push { vec4 p[8]; } pc;

vec3 palette(float t, float mode) {
    vec3 a = vec3(0.04, 0.55, 0.90);
    vec3 b = vec3(1.00, 0.22, 0.06);
    vec3 c = vec3(0.12, 0.88, 0.48);
    return mix(mix(a, b, smoothstep(0.0, 1.0, t)), c, 0.25 + 0.2 * sin(mode));
}

void main() {
    vec2 uv = vUv;
    uint index = min(uint(uv.x * 255.0), 255u);
    vec4 sampleValue = values[index];
    float curve = 1.0 - smoothstep(0.004, 0.018, abs(uv.y - sampleValue.y));
    vec2 grid = abs(fract(uv * vec2(16.0, 9.0)) - 0.5);
    float gridLine = 1.0 - smoothstep(0.47, 0.5, max(grid.x, grid.y));
    float band = smoothstep(0.0, 0.35, sampleValue.y - uv.y) * 0.18;
    vec3 background = mix(vec3(0.008, 0.014, 0.025), vec3(0.018, 0.032, 0.052), uv.y);
    background += gridLine * vec3(0.02, 0.035, 0.055);
    vec3 accent = palette(sampleValue.z, pc.p[0].w);
    vec3 color = background + accent * (curve * 1.35 + band);
    float sweep = exp(-90.0 * abs(uv.x - fract(pc.p[0].z * 0.08)));
    color += accent * sweep * 0.18;
    outColor = vec4(pow(color, vec3(0.4545)), 1.0);
}
