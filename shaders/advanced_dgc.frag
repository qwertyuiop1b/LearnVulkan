#version 450

layout(location = 0) in vec2 quadUv;
layout(location = 1) in vec3 objectColor;
layout(location = 0) out vec4 outColor;

void main() {
    float radius = length(quadUv);
    float core = exp(-radius * radius * 4.0);
    vec3 color = objectColor * (0.25 + core * 1.8);
    float alpha = 1.0 - smoothstep(0.72, 1.0, radius);
    outColor = vec4(color, alpha);
}
