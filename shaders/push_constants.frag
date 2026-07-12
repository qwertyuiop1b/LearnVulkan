#version 450

layout(push_constant) uniform DrawPushConstants {
    float time;
    float offsetX;
    float offsetY;
    float scale;
    float hue;
} pc;

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

vec3 palette(float t) {
    return 0.55 + 0.45 * cos(6.2831853 * (vec3(0.0, 0.33, 0.67) + t));
}

void main() {
    vec3 tint = palette(pc.hue + pc.time * 0.04);
    outColor = vec4(fragColor * tint * 1.4, 1.0);
}
