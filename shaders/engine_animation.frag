#version 450

layout(location = 0) in vec2 vLocal;
layout(location = 1) in vec3 vColor;
layout(location = 0) out vec4 outColor;

void main() {
    float edge = abs(vLocal.x);
    float cap = max(abs(vLocal.y) - 0.82, 0.0) / 0.18;
    float alpha = 1.0 - smoothstep(0.78, 1.0, sqrt(edge*edge + cap*cap));
    float light = 0.35 + 0.65 * sqrt(max(1.0-edge*edge, 0.0));
    outColor = vec4(vColor * light + vec3(0.12) * pow(1.0-edge, 8.0), alpha);
}
