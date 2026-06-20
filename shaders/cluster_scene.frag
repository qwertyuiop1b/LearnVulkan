#version 450
// 第76章：聚簇前向着色 — 片元着色（查找本 cluster 的光源列表）
layout(location=0) in vec3 fragPos;
layout(location=1) in vec3 fragNormal;
layout(location=2) in vec3 fragColor;
layout(location=0) out vec4 outColor;

struct PointLight { vec4 posRadius; vec4 color; };

layout(set=0,binding=1) readonly buffer LightSSBO   { PointLight lights[]; };
layout(set=0,binding=2) readonly buffer ClusterCount{ uint lightCount[]; };
layout(set=0,binding=3) readonly buffer ClusterList { uint lightList[]; };

layout(push_constant) uniform ScenePC {
    mat4  viewProj;
    vec4  cameraPos;
    ivec3 clusterCount;
    int   maxLightsPerCluster;
    vec2  screenSize;
    float nearZ, farZ;
} pc;

uint clusterIndex() {
    // 根据片元的屏幕坐标和深度计算所在 cluster
    vec2 tile     = floor(gl_FragCoord.xy / (pc.screenSize / vec2(pc.clusterCount.xy)));
    float viewZ   = gl_FragCoord.z; // 需要线性化...简化为深度切片
    float logNear = log(pc.nearZ);
    float logFar  = log(pc.farZ);
    float logZ    = log(abs(viewZ));
    uint  zSlice  = uint((logZ - logNear)/(logFar - logNear) * float(pc.clusterCount.z));
    zSlice = clamp(zSlice, 0u, uint(pc.clusterCount.z-1));
    return uint(tile.x) + uint(tile.y)*uint(pc.clusterCount.x)
         + zSlice*uint(pc.clusterCount.x*pc.clusterCount.y);
}

void main() {
    vec3 N  = normalize(fragNormal);
    vec3 V  = normalize(pc.cameraPos.xyz - fragPos);
    vec3 totalLight = fragColor * 0.05;  // ambient

    uint ci  = clusterIndex();
    uint cnt = lightCount[ci];
    for (uint i = 0; i < cnt; ++i) {
        uint li      = lightList[ci * uint(pc.maxLightsPerCluster) + i];
        vec3  lpos   = lights[li].posRadius.xyz;
        float lrad   = lights[li].posRadius.w;
        vec3  lcolor = lights[li].color.rgb * lights[li].color.w;

        vec3  L      = lpos - fragPos;
        float dist   = length(L);
        if (dist > lrad) continue;
        L /= dist;
        float atten  = 1.0 - smoothstep(0.0, lrad, dist);
        float NdotL  = max(dot(N, L), 0.0);
        totalLight  += fragColor * lcolor * NdotL * atten;
    }
    outColor = vec4(totalLight, 1.0);
}
