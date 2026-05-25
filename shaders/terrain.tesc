#version 450
// 第79章：地形 — Tessellation Control Shader（决定细分级别）
layout(vertices=4) out;  // Quad patch

layout(location=0) in  vec2 inXZ[];
layout(location=0) out vec2 outXZ[];

layout(binding=0) uniform TerrainUBO {
    mat4 model, view, proj;
    vec4 cameraPos;
    float terrainSize;
    float heightScale;
    float pad[2];
} ubo;

// 基于与相机距离决定细分级别
float tessLevel(vec2 p0, vec2 p1) {
    vec3 mid  = vec3((p0.x+p1.x)*0.5*ubo.terrainSize, 0.0,
                     (p0.y+p1.y)*0.5*ubo.terrainSize);
    float d   = distance(ubo.cameraPos.xyz, mid);
    float l   = mix(64.0, 1.0, clamp(d/200.0, 0.0, 1.0));
    return max(l, 1.0);
}

void main() {
    outXZ[gl_InvocationID] = inXZ[gl_InvocationID];

    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = tessLevel(inXZ[3], inXZ[0]);
        gl_TessLevelOuter[1] = tessLevel(inXZ[0], inXZ[1]);
        gl_TessLevelOuter[2] = tessLevel(inXZ[1], inXZ[2]);
        gl_TessLevelOuter[3] = tessLevel(inXZ[2], inXZ[3]);
        gl_TessLevelInner[0] = (gl_TessLevelOuter[1]+gl_TessLevelOuter[3])*0.5;
        gl_TessLevelInner[1] = (gl_TessLevelOuter[0]+gl_TessLevelOuter[2])*0.5;
    }
}
