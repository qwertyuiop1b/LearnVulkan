#version 450

struct InstanceData { vec4 positionHeight; vec4 attributes; };
layout(std430, set = 0, binding = 0) readonly buffer Instances { InstanceData data[]; } instances;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;
layout(location = 0) out vec2 bladeUv;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) flat out float variation;

const vec2 QUAD[6] = vec2[](vec2(-1, 0), vec2(1, 0), vec2(1, 1),
                            vec2(-1, 0), vec2(1, 1), vec2(-1, 1));

void main() {
    InstanceData instance = instances.data[gl_InstanceIndex];
    int plane = gl_VertexIndex / 6;
    vec2 corner = QUAD[gl_VertexIndex % 6];
    float yaw = instance.attributes.y + float(plane) * 1.5707963;
    vec3 side = vec3(cos(yaw), 0.0, sin(yaw));
    float vertical = corner.y;
    float wind = sin(pc.p[4].w * 2.4 + instance.attributes.w + dot(instance.positionHeight.xz, vec2(0.11))) *
                 vertical * vertical * pc.p[5].w;
    vec3 root = instance.positionHeight.xyz;
    vec3 position = root + side * corner.x * instance.attributes.x * (1.0 - vertical * 0.72);
    position.y += vertical * instance.positionHeight.w;
    position.xz += vec2(0.72, 0.34) * wind;
    if (instance.positionHeight.w <= 0.0)
        position = vec3(0.0, -10000.0, 0.0);
    mat4 viewProjection = mat4(pc.p[0], pc.p[1], pc.p[2], pc.p[3]);
    gl_Position = viewProjection * vec4(position, 1.0);
    bladeUv = vec2(corner.x * 0.5 + 0.5, vertical);
    worldNormal = normalize(cross(vec3(0.0, 1.0, 0.0) + vec3(wind, 0.0, 0.0), side));
    variation = instance.attributes.z;
}
