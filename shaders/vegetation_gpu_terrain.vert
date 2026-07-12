#version 450

layout(push_constant) uniform SceneParams {
    mat4 viewProjection;
    vec4 cameraTime;   // camera xyz, time
    vec4 terrainWind;  // half extent, height scale, seed, wind strength
} pc;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out float moisture;

const ivec2 CORNERS[6] = ivec2[](
    ivec2(0, 0), ivec2(1, 0), ivec2(1, 1),
    ivec2(0, 0), ivec2(1, 1), ivec2(0, 1));

float hash21(vec2 p, float seed) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32 + seed * 0.013);
    return fract(p.x * p.y);
}

float valueNoise(vec2 p, float seed) {
    vec2 cell = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(cell, seed);
    float b = hash21(cell + vec2(1.0, 0.0), seed);
    float c = hash21(cell + vec2(0.0, 1.0), seed);
    float d = hash21(cell + vec2(1.0, 1.0), seed);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p, float seed) {
    float sum = 0.0;
    float weight = 0.5;
    mat2 rotation = mat2(0.80, -0.60, 0.60, 0.80);
    for (int octave = 0; octave < 5; ++octave) {
        sum += valueNoise(p, seed + float(octave) * 19.7) * weight;
        p = rotation * p * 2.03 + vec2(7.1, -3.8);
        weight *= 0.5;
    }
    return sum / 0.96875;
}

float terrainHeight(vec2 worldXZ) {
    vec2 p = worldXZ * 0.022;
    float broad = fbm(p, pc.terrainWind.z);
    float detail = fbm(p * 2.35 + vec2(11.7, -5.2), pc.terrainWind.z + 41.0);
    float ridge = 1.0 - abs(detail * 2.0 - 1.0);
    return (broad * 0.76 + ridge * 0.24 - 0.43) * pc.terrainWind.y;
}

void main() {
    const int grid = 144;
    const int cells = grid - 1;
    int cellIndex = gl_VertexIndex / 6;
    int cornerIndex = gl_VertexIndex % 6;
    ivec2 cell = ivec2(cellIndex % cells, cellIndex / cells);
    vec2 coordinate = vec2(cell + CORNERS[cornerIndex]) / float(cells);
    vec2 snappedCenter = floor(pc.cameraTime.xz / 2.0) * 2.0;
    vec2 worldXZ = snappedCenter + (coordinate * 2.0 - 1.0) * pc.terrainWind.x;

    float height = terrainHeight(worldXZ);
    float normalStep = pc.terrainWind.x * 2.0 / float(cells);
    float left = terrainHeight(worldXZ - vec2(normalStep, 0.0));
    float right = terrainHeight(worldXZ + vec2(normalStep, 0.0));
    float down = terrainHeight(worldXZ - vec2(0.0, normalStep));
    float up = terrainHeight(worldXZ + vec2(0.0, normalStep));

    worldPosition = vec3(worldXZ.x, height, worldXZ.y);
    worldNormal = normalize(vec3(left - right, normalStep * 2.0, down - up));
    moisture = fbm(worldXZ * 0.018 + vec2(-13.0, 8.0), pc.terrainWind.z + 127.0);
    gl_Position = pc.viewProjection * vec4(worldPosition, 1.0);
}
