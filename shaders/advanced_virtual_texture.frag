#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(std430, set = 0, binding = 0) readonly buffer PageTable { uint entries[]; } pageTable;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    float zoom = pc.p[0].w;
    vec2 virtualUv = (inUV - 0.5) / zoom + pc.p[1].xy;
    virtualUv = fract(virtualUv);
    uvec2 page = min(uvec2(virtualUv * 64.0), uvec2(63));
    uint entry = pageTable.entries[page.x + page.y * 64u];
    bool resident = (entry & 1u) != 0u;
    uint mip = (entry >> 1u) & 7u;
    uint physicalPage = entry >> 4u;
    vec2 localUv = fract(virtualUv * 64.0);
    float frequency = exp2(float(5u - min(mip, 5u)));
    float detail = hash21(floor(localUv * frequency) + vec2(page) * 7.0);
    vec3 terrain = mix(vec3(0.055, 0.16, 0.07), vec3(0.46, 0.38, 0.16), detail);
    terrain = mix(terrain, vec3(0.28, 0.31, 0.34), smoothstep(0.72, 0.92, detail));
    vec3 pageTint = 0.35 + 0.65 * cos(float(physicalPage) * 0.13 + vec3(0.0, 2.1, 4.2));
    vec3 color = resident ? mix(terrain, pageTint, 0.12) : vec3(0.12, 0.015, 0.14);
    float border = max(step(0.965, localUv.x), step(0.965, localUv.y));
    color = mix(color, resident ? vec3(0.03) : vec3(0.85, 0.05, 0.75), border * 0.8);
    float focusRing = 1.0 - smoothstep(0.003, 0.008, abs(length(virtualUv - pc.p[1].xy) - 0.12));
    color += focusRing * vec3(0.55, 0.85, 1.0) * 0.45;
    outColor = vec4(color, 1.0);
}
