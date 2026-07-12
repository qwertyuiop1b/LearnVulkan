#version 450

layout(set = 0, binding = 0) uniform sampler2D terrainData;

layout(push_constant) uniform ClipmapParams {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 terrainParams; // heightScale, terrainScale, baseCellSize, gridResolution
} pc;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 terrainMaterial;
layout(location = 3) flat out uint clipmapLevel;

const ivec2 CORNERS[6] = ivec2[](
    ivec2(0, 0), ivec2(1, 0), ivec2(1, 1),
    ivec2(0, 0), ivec2(1, 1), ivec2(0, 1));

vec4 sampleWorld(vec2 worldXZ) {
    return textureLod(terrainData, worldXZ / pc.terrainParams.y + 0.5, 0.0);
}

void main() {
    int grid = int(pc.terrainParams.w);
    int cells = grid - 1;
    int cellIndex = gl_VertexIndex / 6;
    int cornerIndex = gl_VertexIndex % 6;
    ivec2 cell = ivec2(cellIndex % cells, cellIndex / cells);
    clipmapLevel = uint(gl_InstanceIndex);

    // Higher levels are rings. Remove their center because a denser level covers it.
    int holeMin = grid / 4;
    int holeMax = grid - grid / 4 - 1;
    if (gl_InstanceIndex > 0 && cell.x >= holeMin && cell.x < holeMax &&
        cell.y >= holeMin && cell.y < holeMax) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        worldPosition = vec3(0.0);
        worldNormal = vec3(0, 1, 0);
        terrainMaterial = vec3(0.0);
        return;
    }

    float cellSize = pc.terrainParams.z * exp2(float(gl_InstanceIndex));
    vec2 snappedCenter = floor(pc.cameraPosition.xz / cellSize) * cellSize;
    vec2 gridCoordinate = vec2(cell + CORNERS[cornerIndex]) - vec2(float(grid - 1) * 0.5);
    vec2 worldXZ = snappedCenter + gridCoordinate * cellSize;
    vec4 terrain = sampleWorld(worldXZ);
    float height = terrain.r * pc.terrainParams.x;

    float sampleStep = pc.terrainParams.y / float(textureSize(terrainData, 0).x);
    float heightRight = sampleWorld(worldXZ + vec2(sampleStep, 0)).r * pc.terrainParams.x;
    float heightUp = sampleWorld(worldXZ + vec2(0, sampleStep)).r * pc.terrainParams.x;
    worldNormal = normalize(vec3(height - heightRight, sampleStep, height - heightUp));
    worldPosition = vec3(worldXZ.x, height, worldXZ.y);
    terrainMaterial = terrain.rgb;
    gl_Position = pc.viewProjection * vec4(worldPosition, 1.0);
}
