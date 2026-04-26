#version 450
// 延迟光照通道顶点着色器（第19章）
// 渲染全屏四边形，对每个像素执行光照计算

// 全屏三角形（无顶点缓冲，直接从 gl_VertexIndex 生成坐标）
void main()
{
    // 用 gl_VertexIndex 生成覆盖整个屏幕的大三角形
    // VertexIndex: 0→(-1,-1), 1→(3,-1), 2→(-1,3)
    vec2 pos = vec2((gl_VertexIndex & 1) << 2, (gl_VertexIndex & 2) << 1) - 1.0;
    gl_Position = vec4(pos, 0.0, 1.0);
}
