#version 450
// 第79章：地形 LOD — 顶点着色器（输出 patch 顶点给 tesc）
layout(location=0) in vec2 inXZ;    // patch 的 xz 坐标（归一化）

layout(binding=0) uniform TerrainUBO {
    mat4 model, view, proj;
    vec4 cameraPos;
    float terrainSize;
    float heightScale;
    float pad[2];
} ubo;

layout(location=0) out vec2 outXZ;

void main() {
    outXZ = inXZ;
    // 高度将在 tese 中从高度图采样
    vec4 pos = ubo.model * vec4(inXZ.x*ubo.terrainSize, 0.0, inXZ.y*ubo.terrainSize, 1.0);
    gl_Position = ubo.proj * ubo.view * pos;
}
