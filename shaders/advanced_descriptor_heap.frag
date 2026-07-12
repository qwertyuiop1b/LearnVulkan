#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(std430, set = 0, binding = 0) readonly buffer MaterialHeap { vec4 materials[]; } heap;
layout(std430, set = 0, binding = 1) readonly buffer ObjectMaterials { uint materialIndex[]; } objects;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;

void main() {
    uvec2 grid = uvec2(32, 18);
    uvec2 cell = min(uvec2(inUV * vec2(grid)), grid - uvec2(1));
    uint objectIndex = cell.x + cell.y * grid.x;
    uint materialIndex = objects.materialIndex[objectIndex];
    vec4 material = heap.materials[materialIndex];
    vec2 local = fract(inUV * vec2(grid)) * 2.0 - 1.0;
    float radiusSquared = dot(local, local);
    vec3 background = vec3(0.012, 0.018, 0.032);
    if (radiusSquared > 0.82) {
        outColor = vec4(background, 1.0);
        return;
    }
    vec3 normal = normalize(vec3(local, sqrt(max(1.0 - radiusSquared, 0.0))));
    vec3 light = normalize(vec3(-0.45, 0.72, 0.53));
    vec3 view = vec3(0.0, 0.0, 1.0);
    float diffuse = max(dot(normal, light), 0.0);
    float exponent = mix(96.0, 5.0, material.w);
    float specular = pow(max(dot(reflect(-light, normal), view), 0.0), exponent);
    vec3 color = material.rgb * (0.12 + diffuse * 0.92) + specular * (1.0 - material.w) * 0.8;
    float selected = step(float(materialIndex), pc.p[0].z) * step(pc.p[0].z, float(materialIndex));
    color += selected * vec3(0.8, 0.5, 0.05);
    outColor = vec4(color, 1.0);
}
