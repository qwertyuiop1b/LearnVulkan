#version 450
// VRS 演示顶点着色器
layout(location = 0) out vec2 fragCoord;
void main() {
    vec2 pos = vec2((gl_VertexIndex & 1) << 2, (gl_VertexIndex & 2) << 1) - 1.0;
    gl_Position = vec4(pos, 0.0, 1.0);
    fragCoord = pos;
}
