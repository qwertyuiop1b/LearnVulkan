#version 450
// 第79章：地形片元着色（高度着色 + 法线光照）
layout(location=0) in vec3 fragPos;
layout(location=1) in vec3 fragNormal;
layout(location=2) in vec2 fragUV;
layout(location=0) out vec4 outColor;

layout(push_constant) uniform TerrainFPC {
    vec4 lightDir;
    float heightScale;
    float pad[3];
} pc;

void main() {
    vec3 N   = normalize(fragNormal);
    float h  = fragPos.y / pc.heightScale;   // 归一化高度 [0,1]

    // 高度分层着色
    vec3 snow  = vec3(0.95, 0.95, 0.98);
    vec3 rock  = vec3(0.45, 0.40, 0.35);
    vec3 grass = vec3(0.30, 0.55, 0.20);
    vec3 sand  = vec3(0.76, 0.70, 0.50);

    vec3 baseColor;
    if      (h > 0.80) baseColor = mix(rock, snow,  smoothstep(0.80, 0.95, h));
    else if (h > 0.50) baseColor = mix(grass,rock,  smoothstep(0.50, 0.80, h));
    else if (h > 0.15) baseColor = mix(sand, grass, smoothstep(0.15, 0.50, h));
    else               baseColor = sand;

    float NdotL = max(dot(N, normalize(-pc.lightDir.xyz)), 0.0);
    outColor = vec4(baseColor * (0.2 + NdotL * 0.8), 1.0);
}
