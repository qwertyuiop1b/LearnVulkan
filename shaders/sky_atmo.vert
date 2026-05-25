#version 450
// 第73章：大气散射 — 全屏顶点（等经纬映射）
layout(location=0) out vec2 outUV;
void main() {
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
