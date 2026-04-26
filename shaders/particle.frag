#version 450
// 粒子片段着色器
// 将点渲染为圆形（通过内置变量 gl_PointCoord 实现）

layout(location = 0) in  vec4 fragColor;
layout(location = 0) out vec4 outColor;

void main()
{
    // gl_PointCoord: 当前像素在点内的位置，范围 [0,1]x[0,1]
    // 中心点坐标为 (0.5, 0.5)
    // 计算到中心的距离，超过 0.5 则丢弃（形成圆形）
    vec2  coord = gl_PointCoord - vec2(0.5);
    float dist  = length(coord);
    if (dist > 0.5) discard;   // 点的角落丢弃，变成圆形粒子

    // 边缘渐变（柔化边缘）
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    outColor = vec4(fragColor.rgb, fragColor.a * alpha);
}
