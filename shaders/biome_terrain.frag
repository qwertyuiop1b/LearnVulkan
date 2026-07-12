#version 450

layout(push_constant) uniform ClipmapParams {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 terrainParams;
} pc;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec3 terrainMaterial; // height, moisture, rock mask
layout(location = 3) flat in uint clipmapLevel;
layout(location = 0) out vec4 outColor;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    vec3 normal = normalize(worldNormal);
    float elevation = clamp(worldPosition.y / pc.terrainParams.x, 0.0, 1.0);
    float moisture = terrainMaterial.g;
    float slope = 1.0 - normal.y;
    float latitude = abs(worldPosition.z) / (pc.terrainParams.y * 0.5);
    float temperature = clamp(1.05 - latitude * 0.58 - elevation * 0.72, 0.0, 1.0);

    float desertWeight = smoothstep(0.56, 0.78, temperature) * (1.0 - smoothstep(0.25, 0.48, moisture));
    float forestWeight = smoothstep(0.38, 0.62, moisture) * smoothstep(0.24, 0.48, temperature) *
                         (1.0 - smoothstep(0.78, 0.94, temperature));
    float snowWeight = smoothstep(0.72, 0.91, elevation) + smoothstep(0.72, 0.96, latitude) * 0.35;
    float rockWeight = clamp(slope * 3.1 + terrainMaterial.b * 0.55, 0.0, 1.0) * (1.0 - snowWeight * 0.7);
    float grassWeight = max(0.0, 1.0 - desertWeight - forestWeight - snowWeight - rockWeight);

    float total = max(desertWeight + forestWeight + snowWeight + rockWeight + grassWeight, 0.001);
    vec3 desert = vec3(0.57, 0.39, 0.16);
    vec3 grassland = vec3(0.22, 0.38, 0.095);
    vec3 forest = vec3(0.035, 0.19, 0.075);
    vec3 rock = vec3(0.30, 0.29, 0.28);
    vec3 snow = vec3(0.84, 0.89, 0.91);
    vec3 base = (desert * desertWeight + grassland * grassWeight + forest * forestWeight +
                 rock * rockWeight + snow * snowWeight) / total;

    float detail = hash21(floor(worldPosition.xz * 1.7));
    base *= mix(0.88, 1.10, detail);
    vec3 sun = normalize(vec3(-0.42, 0.82, -0.38));
    float diffuse = max(dot(normal, sun), 0.0);
    float distanceToCamera = distance(worldPosition, pc.cameraPosition.xyz);
    float fog = 1.0 - exp(-distanceToCamera * 0.0035);
    vec3 color = base * (0.22 + diffuse * 0.88);
    color = mix(color, vec3(0.49, 0.64, 0.77), clamp(fog, 0.0, 0.74));
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
