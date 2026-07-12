#version 450

struct VegetationInstance {
    vec4 positionHeight;
    vec4 attributes;
};

layout(std430, set = 0, binding = 0) readonly buffer InstanceBuffer {
    VegetationInstance instances[];
};

layout(push_constant) uniform SceneParams {
    mat4 viewProjection;
    vec4 cameraTime;
    vec4 terrainWind;
} pc;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 bladeUv;
layout(location = 3) flat out float colorVariation;

const vec2 QUAD_CORNERS[6] = vec2[](
    vec2(-1.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    vec2(-1.0, 0.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));

void main() {
    VegetationInstance instance = instances[gl_InstanceIndex];
    uint vertex = uint(gl_VertexIndex);
    uint plane = vertex / 18u;
    uint localVertex = vertex % 18u;
    uint segment = localVertex / 6u;
    uint corner = localVertex % 6u;
    vec2 quad = QUAD_CORNERS[corner];

    float segmentStart = float(segment) / 3.0;
    float t = (float(segment) + quad.y) / 3.0;
    float yaw = instance.attributes.y + float(plane) * 1.5707963;
    vec3 bladeRight = vec3(cos(yaw), 0.0, sin(yaw));
    vec3 bladeNormal = normalize(vec3(-bladeRight.z, 0.22, bladeRight.x));

    float gust = sin(pc.cameraTime.w * 1.75 + instance.attributes.w +
                     instance.positionHeight.x * 0.073 + instance.positionHeight.z * 0.051);
    vec3 windDirection = normalize(vec3(0.82, 0.0, 0.42));
    float bend = t * t * pc.terrainWind.w * (0.30 + gust * 0.17) * instance.positionHeight.w;
    float taper = mix(1.0, 0.10, t);
    float width = instance.attributes.x * taper;

    vec3 root = instance.positionHeight.xyz;
    vec3 world = root + vec3(0.0, t * instance.positionHeight.w, 0.0);
    world += bladeRight * quad.x * width;
    world += windDirection * bend;

    worldPosition = world;
    worldNormal = normalize(mix(bladeNormal, vec3(-windDirection.x, 0.55, -windDirection.z), t * 0.35));
    bladeUv = vec2(quad.x * 0.5 + 0.5, segmentStart + quad.y / 3.0);
    colorVariation = instance.attributes.z;
    gl_Position = pc.viewProjection * vec4(world, 1.0);
}
