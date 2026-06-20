#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sceneTex;
layout(set = 0, binding = 1) uniform sampler2D depthTex;

layout(push_constant) uniform FogPC {
    mat4  invProjView;       // 用于从深度重建世界坐标
    vec4  cameraPos;
    vec4  fogColor;
    vec4  sunDir;            // xyz=方向, w=god rays 强度
    float fogDensity;
    float fogHeightFalloff;  // 高度衰减系数
    float fogStart;          // 水平雾起始距离
    int   enableGodRays;
} pc;

// 指数高度雾密度积分（沿视线方向的解析解）
float heightFogDensity(vec3 worldPos, vec3 camPos)
{
    float h      = worldPos.y - camPos.y;
    float dist   = length(worldPos - camPos);
    float falloff = pc.fogHeightFalloff;
    // 高度雾：density = fogDensity * exp(-falloff * y)
    // 沿射线积分近似
    float density = pc.fogDensity * dist * exp(-falloff * max(worldPos.y, 0.0));
    return clamp(density, 0.0, 1.0);
}

// God Rays（Radial 模糊近似）
vec3 godRays(vec2 uv, vec3 sceneColor)
{
    // 把太阳方向投影到屏幕
    vec4 sunClip = vec4(-pc.sunDir.xyz, 1.0);
    // 简化：假设光源在屏幕外侧（偏上方）
    vec2 sunPos  = vec2(0.5, 0.9);
    vec2 dir     = uv - sunPos;
    float dist   = length(dir);

    vec3 rays = vec3(0.0);
    const int SAMPLES = 8;
    float decay = 0.92;
    float weight = 0.04;
    float d = decay;
    for (int i = 0; i < SAMPLES; ++i) {
        vec2 sampleUV = uv - dir * (float(i) / float(SAMPLES)) * 0.3;
        sampleUV = clamp(sampleUV, 0.001, 0.999);
        float sceneDepth = texture(depthTex, sampleUV).r;
        vec3  col = texture(sceneTex, sampleUV).rgb;
        // 天空区域（depth=1）贡献光
        float skyMask = step(0.999, sceneDepth);
        rays += col * skyMask * weight * d;
        d *= decay;
    }
    return rays * pc.sunDir.w * (1.0 - clamp(dist * 2.0, 0.0, 1.0));
}

void main()
{
    vec3 sceneColor = texture(sceneTex, inUV).rgb;

    // 从深度重建世界坐标
    float depth = texture(depthTex, inUV).r;
    vec4 ndcPos  = vec4(inUV * 2.0 - 1.0, depth, 1.0);
    vec4 worldH  = pc.invProjView * ndcPos;
    vec3 worldPos = worldH.xyz / worldH.w;

    // 高度雾
    float fogFactor = 0.0;
    if (depth < 1.0) {
        fogFactor = heightFogDensity(worldPos, pc.cameraPos.xyz);
        fogFactor = 1.0 - exp(-fogFactor * 3.0);
    }

    vec3 color = mix(sceneColor, pc.fogColor.rgb, fogFactor);

    // God Rays（可选）
    if (pc.enableGodRays != 0) {
        color += godRays(inUV, sceneColor);
    }

    outColor = vec4(color, 1.0);
}
