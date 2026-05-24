#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

void main()
{
    // 程序化软圆形精灵（无需纹理文件）
    vec2 uv    = fragUV * 2.0 - 1.0;
    float dist = length(uv);

    // 中心亮，边缘渐淡（软粒子，超出范围 alpha→0 而不是 discard，避免 DemoteToHelperInvocation）
    float alpha = (1.0 - smoothstep(0.6, 1.0, dist)) * fragColor.a;
    // 中心更亮（火焰感）
    float glow  = 1.0 - dist * 0.7;

    outColor = vec4(fragColor.rgb * glow, alpha);
}
