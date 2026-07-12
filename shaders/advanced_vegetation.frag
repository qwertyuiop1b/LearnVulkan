#version 450

layout(location = 0) in vec2 bladeUv;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) flat in float variation;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(gl_FrontFacing ? worldNormal : -worldNormal);
    vec3 base = mix(vec3(0.025, 0.11, 0.02), vec3(0.18, 0.42, 0.045), bladeUv.y);
    base *= mix(0.78, 1.24, variation);
    float diffuse = max(dot(normal, normalize(vec3(-0.42, 0.84, -0.34))), 0.0);
    outColor = vec4(base * (0.28 + diffuse * 0.94), 1.0);
}
