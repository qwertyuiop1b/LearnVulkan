#version 450
// 粒子顶点着色器
// 粒子被渲染为点（GL_POINTS / POINT_LIST），
// gl_PointSize 控制每个点的像素大小。

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inVelocity;   // 计算着色器写入的速度（这里用来给点上色）
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec4 fragColor;

void main()
{
    gl_Position  = vec4(inPosition, 0.0, 1.0);
    gl_PointSize = 2.0;   // 每个粒子 2×2 像素

    // 根据速度大小调整亮度（速度越快颜色越亮）
    float speed = length(inVelocity);
    fragColor = vec4(inColor.rgb * (0.5 + speed * 2.0), inColor.a);
}
