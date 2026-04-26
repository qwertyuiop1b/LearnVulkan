#version 450
// ═══════════════════════════════════════════════════════════════════════════
// 片段着色器：彩色三角形
//
// 每个光栅化后的像素（片段）调用一次
// fragColor 是从顶点着色器插值得到的颜色
// ═══════════════════════════════════════════════════════════════════════════

// layout(location = N) in：接收顶点着色器输出的插值颜色
layout(location = 0) in vec3 fragColor;

// layout(location = N) out：输出到帧缓冲的颜色（RGBA）
layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(fragColor, 1.0);   // alpha = 1.0（不透明）
}
