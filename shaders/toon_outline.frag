#version 450
// 第84章：卡通渲染 — 轮廓线颜色

layout(location=0) out vec4 outColor;

layout(push_constant) uniform OutlinePC {
    float outlineWidth;
    float outlineWidthNDC;
    int   useNDCWidth;
    float pad1;
    vec4  outlineColor;
} pc;

void main()
{
    outColor = pc.outlineColor;
}
