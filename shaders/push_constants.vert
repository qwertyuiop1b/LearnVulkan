#version 450

layout(push_constant) uniform DrawPushConstants {
    float time;
    float offsetX;
    float offsetY;
    float scale;
    float hue;
} pc;

layout(location = 0) out vec3 fragColor;

const vec2 positions[3] = vec2[](vec2(0.0, -0.58), vec2(0.52, 0.42), vec2(-0.52, 0.42));
const vec3 colors[3] = vec3[](vec3(1.0, 0.18, 0.12), vec3(0.1, 0.9, 0.45), vec3(0.12, 0.35, 1.0));

void main() {
    float angle = pc.time * (0.35 + pc.hue * 0.5);
    mat2 rotation = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
    vec2 p = rotation * positions[gl_VertexIndex] * pc.scale + vec2(pc.offsetX, pc.offsetY);
    gl_Position = vec4(p, 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}
