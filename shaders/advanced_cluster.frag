#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(std430, set = 0, binding = 0) readonly buffer ClusterCounts { uint counts[]; } clusterCounts;
layout(std430, set = 0, binding = 1) readonly buffer ClusterLists { uint indices[]; } clusterLists;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;

vec4 lightData(uint index) {
    float fi = float(index);
    float angle = fi * 2.39996323 + pc.p[0].z * (0.18 + mod(fi, 5.0) * 0.025);
    float ring = 0.12 + 0.36 * fract(fi * 0.618034);
    vec2 screen = vec2(0.5) + vec2(cos(angle), sin(angle * 1.13)) * ring;
    float depth = 0.04 + 0.92 * fract(fi * 0.75487766);
    return vec4(screen, depth, 0.08 + 0.08 * fract(fi * 0.37));
}

vec3 lightColor(uint index) {
    return 0.45 + 0.55 * cos(float(index) * 1.71 + vec3(0.0, 2.1, 4.2));
}

void main() {
    uvec3 dimensions = uvec3(pc.p[1].xyz + 0.5);
    uint maxLights = uint(pc.p[1].w + 0.5);
    vec2 p = inUV * 2.0 - 1.0;
    float ripple = sin(p.x * 8.0) * cos(p.y * 7.0) * 0.06;
    float depth = clamp(length(p) * 0.72 + 0.16 + ripple, 0.0, 0.999);
    vec3 normal = normalize(vec3(-cos(p.x * 8.0) * 0.35, -sin(p.y * 7.0) * 0.35, 1.0));
    uvec2 tile = min(uvec2(inUV * vec2(dimensions.xy)), dimensions.xy - uvec2(1));
    uint slice = min(uint(depth * float(dimensions.z)), dimensions.z - 1u);
    uint clusterIndex = tile.x + tile.y * dimensions.x + slice * dimensions.x * dimensions.y;
    uint count = clusterCounts.counts[clusterIndex];
    vec3 base = mix(vec3(0.025, 0.045, 0.075), vec3(0.12, 0.16, 0.20), 1.0 - depth);
    vec3 color = base * 0.24;
    for (uint listIndex = 0u; listIndex < count; ++listIndex) {
        uint lightIndex = clusterLists.indices[clusterIndex * maxLights + listIndex];
        vec4 light = lightData(lightIndex);
        vec2 delta = light.xy - inUV;
        float distanceToLight = length(delta);
        float attenuation = pow(max(0.0, 1.0 - distanceToLight / light.w), 2.0);
        vec3 lightDirection = normalize(vec3(delta * 4.0, light.z - depth));
        color += lightColor(lightIndex) * max(dot(normal, lightDirection), 0.0) * attenuation * 1.8;
    }
    float clusterEdge = max(step(0.985, fract(inUV.x * float(dimensions.x))),
                            step(0.985, fract(inUV.y * float(dimensions.y))));
    color = mix(color, vec3(0.08, 0.16, 0.22), clusterEdge * 0.24);
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
