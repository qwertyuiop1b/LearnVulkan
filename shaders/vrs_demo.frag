#version 450
// VRS（Variable Rate Shading）演示片段着色器（第29章）
// 根据片段颜色展示着色率区域

layout(location = 0) in  vec2 fragCoord;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    vec2  resolution;
    float time;
} pc;

void main()
{
    vec2 uv = gl_FragCoord.xy / pc.resolution;

    // 可视化 VRS 区域：不同区域使用不同着色率
    // 屏幕中央（焦点区域）= 1x1（全精度）
    // 周边区域           = 2x2 或 4x4（低精度）
    vec2  center  = vec2(0.5);
    float distFromCenter = length(uv - center);

    // 着色率可视化（仅示意，真实 VRS 由 shading rate attachment 控制）
    vec3 color;
    if (distFromCenter < 0.15) {
        color = vec3(0.2, 0.8, 0.2);  // 绿色=1x1 全精度区域
    } else if (distFromCenter < 0.35) {
        color = vec3(0.8, 0.8, 0.2);  // 黄色=2x2 中等精度
    } else {
        color = vec3(0.8, 0.2, 0.2);  // 红色=4x4 低精度边缘
    }

    // 叠加波纹图案（显示"在低精度区域你会漏掉多少细节"）
    float wave = 0.5 + 0.5 * sin(uv.x * 50.0 + pc.time) * sin(uv.y * 50.0 + pc.time);
    color = mix(color, vec3(wave), 0.3);

    outColor = vec4(color, 1.0);
}
