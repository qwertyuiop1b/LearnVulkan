#version 450

// 全屏三角形技巧：不需要顶点缓冲区，gl_VertexIndex 生成坐标
// Draw(3, 1, 0, 0) 即可覆盖整个屏幕

layout(location = 0) out vec2 outUV;

void main()
{
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
