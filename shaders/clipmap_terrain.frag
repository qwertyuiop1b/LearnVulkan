#version 450

layout(push_constant) uniform ClipmapParams {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 terrainParams;
} pc;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec3 terrainMaterial;
layout(location = 3) flat in uint clipmapLevel;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(worldNormal);
    float height = worldPosition.y / pc.terrainParams.x;
    float slope = 1.0 - normal.y;
    float moisture = terrainMaterial.g;

    vec3 sand = vec3(0.43, 0.35, 0.20);
    vec3 dryGrass = vec3(0.20, 0.34, 0.10);
    vec3 wetGrass = vec3(0.055, 0.21, 0.10);
    vec3 rock = vec3(0.29, 0.30, 0.29);
    vec3 snow = vec3(0.78, 0.84, 0.87);
    vec3 base = mix(sand, mix(dryGrass, wetGrass, moisture), smoothstep(0.16, 0.34, height));
    base = mix(base, rock, clamp(slope * 2.8 + terrainMaterial.b * 0.45, 0.0, 1.0));
    base = mix(base, snow, smoothstep(0.76, 0.90, height));

    vec3 lightDirection = normalize(vec3(-0.42, 0.82, -0.38));
    float diffuse = max(dot(normal, lightDirection), 0.0);
    float distanceToCamera = distance(worldPosition, pc.cameraPosition.xyz);
    float fog = 1.0 - exp(-distanceToCamera * 0.0038);
    vec3 sky = vec3(0.48, 0.63, 0.76);
    vec3 color = base * (0.22 + diffuse * 0.88);

    // A very subtle level tint makes LOD rings inspectable without dominating the material.
    vec3 levelTint = vec3(0.015, 0.012, 0.008) * float(clipmapLevel);
    color = mix(color + levelTint, sky, clamp(fog, 0.0, 0.76));
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
