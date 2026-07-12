#version 450

layout(push_constant) uniform WorldPush {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 params;   // time, mode, reserved, terrainScale
    vec4 controls; // river width, road width, channel depth, reserved
} pc;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 surfaceUV;
layout(location = 3) flat out uint materialId;

const ivec2 TRI_CORNERS[6] = ivec2[](
    ivec2(0, 0), ivec2(1, 0), ivec2(1, 1),
    ivec2(0, 0), ivec2(1, 1), ivec2(0, 1));
const vec2 RIBBON_CORNERS[6] = vec2[](
    vec2(0, -1), vec2(0, 1), vec2(1, 1),
    vec2(0, -1), vec2(1, 1), vec2(1, -1));

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise2(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x),
               mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), f.x), f.y) * 2.0 - 1.0;
}

float riverCenter(float z) {
    return sin(z * 0.075) * 8.0 + sin(z * 0.21 + 1.3) * 2.2;
}

float roadCenter(float z) {
    return -16.0 + sin(z * 0.052 - 0.8) * 5.5 + sin(z * 0.14) * 1.3;
}

float baseTerrainHeight(vec2 xz) {
    float h = noise2(xz * 0.045) * 3.4 + noise2(xz * 0.11 + 17.0) * 1.2;
    h += sin(xz.x * 0.035) * cos(xz.y * 0.04) * 2.1;
    return h;
}

float riverBedHeight(float z) {
    return baseTerrainHeight(vec2(riverCenter(z), z)) - pc.controls.z;
}

float terrainHeight(vec2 xz) {
    float h = baseTerrainHeight(xz);
    float riverDistance = abs(xz.x - riverCenter(xz.y));
    float halfWidth = max(pc.controls.x * 0.5, 0.5);
    float channel = 1.0 - smoothstep(halfWidth * 1.05, halfWidth * 1.75, riverDistance);
    return mix(h, riverBedHeight(xz.y), channel);
}

vec3 terrainNormal(vec2 xz) {
    float stepSize = 0.18;
    float h = terrainHeight(xz);
    return normalize(vec3(h - terrainHeight(xz + vec2(stepSize, 0)), stepSize,
                          h - terrainHeight(xz + vec2(0, stepSize))));
}

void main() {
    uint mode = uint(pc.params.y + 0.5);
    if (mode == 0u) {
        const int GRID = 112;
        int cells = GRID - 1;
        int cellIndex = gl_VertexIndex / 6;
        ivec2 cell = ivec2(cellIndex % cells, cellIndex / cells);
        ivec2 corner = TRI_CORNERS[gl_VertexIndex % 6];
        vec2 normalized = vec2(cell + corner) / float(cells) - 0.5;
        vec2 xz = normalized * pc.params.w;
        worldPosition = vec3(xz.x, terrainHeight(xz), xz.y);
        worldNormal = terrainNormal(xz);
        surfaceUV = normalized + 0.5;
        materialId = 0u;
    } else {
        const int SEGMENTS = 192;
        int segment = gl_VertexIndex / 6;
        vec2 corner = RIBBON_CORNERS[gl_VertexIndex % 6];
        float t = (float(segment) + corner.x) / float(SEGMENTS);
        float z = mix(-pc.params.w * 0.48, pc.params.w * 0.48, t);
        float centerX = mode == 1u ? riverCenter(z) : roadCenter(z);
        float nextZ = z + 0.15;
        float nextX = mode == 1u ? riverCenter(nextZ) : roadCenter(nextZ);
        vec2 tangent = normalize(vec2(nextX - centerX, nextZ - z));
        vec2 side = vec2(-tangent.y, tangent.x);
        float ribbonWidth = mode == 1u ? pc.controls.x : pc.controls.y;
        float halfWidth = ribbonWidth * 0.5;
        vec2 xz = vec2(centerX, z) + side * corner.y * halfWidth;
        float ground = terrainHeight(xz);
        float surfaceHeight = mode == 1u ? riverBedHeight(xz.y) + 0.18 : ground + 0.09;
        worldPosition = vec3(xz.x, surfaceHeight, xz.y);
        worldNormal = mode == 1u ? vec3(0, 1, 0) : terrainNormal(xz);
        surfaceUV = vec2(t * 18.0, corner.y * 0.5 + 0.5);
        materialId = mode;
    }
    gl_Position = pc.viewProjection * vec4(worldPosition, 1.0);
}
