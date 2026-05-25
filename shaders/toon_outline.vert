#version 450
/**
 * 第84章：卡通渲染 — 轮廓线顶点着色器
 *
 * 技术：Inverted Hull（反转法线外扩）
 *   做法：将顶点沿法线方向向外偏移一小段距离，
 *         同时翻转剔除面（只绘制背面），
 *         这样从正面看时背后多出来的一圈就是轮廓线。
 *
 * 优点：简单高效，不需要额外 Pass 分析边缘
 * 缺点：在硬边（法线不连续）处可能有裂缝
 */

layout(location=0) in vec3 inPosition;
layout(location=1) in vec3 inNormal;
layout(location=2) in vec3 inColor;

layout(binding=0) uniform ToonUBO {
    mat4 model, view, proj;
    vec4 lightDir;
    vec4 lightColor;
    vec4 cameraPos;
    float specularSize;
    float rimPower;
    float pad[2];
} ubo;

layout(push_constant) uniform OutlinePC {
    float outlineWidth;    // 轮廓线宽度（世界空间单位）
    float outlineWidthNDC; // NDC 空间的固定宽度（屏幕空间均匀）
    int   useNDCWidth;     // 0=世界空间, 1=屏幕空间（推荐）
    float pad;
} pc;

void main()
{
    vec3 posWorld  = (ubo.model * vec4(inPosition, 1.0)).xyz;
    vec3 normWorld = normalize(mat3(transpose(inverse(ubo.model))) * inNormal);

    vec4 posClip   = ubo.proj * ubo.view * vec4(posWorld, 1.0);

    if (pc.useNDCWidth != 0) {
        // ── 屏幕空间均匀宽度（推荐）──────────────────────────────
        // 将法线变换到 NDC 空间，按屏幕像素均匀偏移
        vec4 normClip  = ubo.proj * ubo.view * vec4(normWorld, 0.0);
        vec2 normNDC   = normalize(normClip.xy);
        posClip.xy    += normNDC * pc.outlineWidthNDC * posClip.w;
    } else {
        // ── 世界空间宽度（简单但近大远小）─────────────────────────
        posWorld  += normWorld * pc.outlineWidth;
        posClip    = ubo.proj * ubo.view * vec4(posWorld, 1.0);
    }

    gl_Position = posClip;
}
