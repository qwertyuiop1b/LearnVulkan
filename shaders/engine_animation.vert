#version 450

struct Bone { vec4 aRadius; vec4 bKind; vec4 color; };
layout(set = 0, binding = 0, std430) readonly buffer Skeleton { Bone bones[]; };
layout(push_constant) uniform Push { vec4 p[8]; } pc;
layout(location = 0) out vec2 vLocal;
layout(location = 1) out vec3 vColor;

void main() {
    const vec2 corners[6] = vec2[6](vec2(-1,0),vec2(1,0),vec2(1,1),vec2(-1,0),vec2(1,1),vec2(-1,1));
    Bone b = bones[gl_InstanceIndex];
    vec2 corner = corners[gl_VertexIndex];
    vec3 center = mix(b.aRadius.xyz, b.bKind.xyz, corner.y);
    vec3 axis = normalize(b.bKind.xyz - b.aRadius.xyz);
    vec3 view = normalize(pc.p[4].xyz - center);
    vec3 side = normalize(cross(axis, view));
    vec3 world = center + side * corner.x * b.aRadius.w;
    gl_Position = mat4(pc.p[0],pc.p[1],pc.p[2],pc.p[3]) * vec4(world,1);
    vLocal = vec2(corner.x, corner.y * 2.0 - 1.0);
    vColor = b.color.rgb;
}
