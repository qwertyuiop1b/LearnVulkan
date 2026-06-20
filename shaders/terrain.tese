#version 450
// 第79章：地形 — Tessellation Evaluation Shader（采样高度图）
layout(quads, fractional_odd_spacing, ccw) in;

layout(location=0) in  vec2 inXZ[];
layout(location=0) out vec3 fragPos;
layout(location=1) out vec3 fragNormal;
layout(location=2) out vec2 fragUV;

layout(binding=0) uniform TerrainUBO {
    mat4 model, view, proj;
    vec4 cameraPos;
    float terrainSize;
    float heightScale;
    float pad[2];
} ubo;

layout(set=0,binding=1) uniform sampler2D heightMap;

float getHeight(vec2 uv) {
    return texture(heightMap, uv).r * ubo.heightScale;
}

void main() {
    vec2 uv0 = mix(inXZ[0], inXZ[1], gl_TessCoord.x);
    vec2 uv1 = mix(inXZ[3], inXZ[2], gl_TessCoord.x);
    vec2 uv  = mix(uv0, uv1, gl_TessCoord.y);

    float h  = getHeight(uv);

    // 有限差分法计算法线
    float d  = 1.0 / textureSize(heightMap, 0).x;
    float hR = getHeight(uv + vec2(d, 0));
    float hU = getHeight(uv + vec2(0, d));
    vec3 N   = normalize(vec3(h-hR, d*ubo.terrainSize, h-hU));

    vec3 pos = vec3(uv.x*ubo.terrainSize, h, uv.y*ubo.terrainSize);
    fragPos    = (ubo.model * vec4(pos, 1.0)).xyz;
    fragNormal = N;
    fragUV     = uv;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(pos, 1.0);
}
