#version 450

/**
 * 全屏贴花叠加（Fullscreen Decal Overlay）
 *
 * 做法：对每个像素从深度重建世界坐标，检测是否在各贴花 box 内，
 *       若在则按贴花颜色 Alpha 混合叠加到场景颜色上。
 * 优点：无需 box 几何、无需每贴花 draw call、无 push constant 超限问题。
 */

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sceneTex;
layout(set = 0, binding = 1) uniform sampler2D depthTex;

struct DecalData {
    mat4  invWorld;    // 世界坐标 → 贴花局部空间
    vec4  color;       // rgba（a 为最大不透明度）
};

layout(set = 0, binding = 2) uniform DecalUBO {
    mat4      invProjView;
    int       decalCount;
    float     pad[3];
    DecalData decals[8];
} ubo;

void main()
{
    vec3 sceneColor = texture(sceneTex, inUV).rgb;
    float depth     = texture(depthTex,  inUV).r;

    if (depth >= 1.0) {
        outColor = vec4(sceneColor, 1.0);
        return;
    }

    // 从深度重建世界坐标
    vec4 ndcPos   = vec4(inUV * 2.0 - 1.0, depth, 1.0);
    vec4 worldH   = ubo.invProjView * ndcPos;
    vec3 worldPos = worldH.xyz / worldH.w;

    // 累加所有贴花影响
    vec3 finalColor = sceneColor;
    for (int i = 0; i < ubo.decalCount; ++i) {
        // 变换到贴花局部空间 [-0.5, 0.5]^3
        vec4 localPos = ubo.decals[i].invWorld * vec4(worldPos, 1.0);
        vec3 local    = localPos.xyz;

        if (any(greaterThan(abs(local), vec3(0.5)))) continue;

        // 边缘渐变 Alpha
        vec2 fade  = 1.0 - smoothstep(0.35, 0.5, abs(local.xz));
        float yFade= 1.0 - smoothstep(0.3, 0.5, abs(local.y));
        float alpha= fade.x * fade.y * yFade * ubo.decals[i].color.a;

        finalColor = mix(finalColor, ubo.decals[i].color.rgb, alpha);
    }

    outColor = vec4(finalColor, 1.0);
}
