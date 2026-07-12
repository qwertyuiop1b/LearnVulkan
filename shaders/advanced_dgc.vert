#version 450

struct ObjectData { vec4 positionScale; vec4 color; };
layout(std430, set = 0, binding = 0) readonly buffer Objects { ObjectData data[]; } objects;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;
layout(location = 0) out vec2 quadUv;
layout(location = 1) out vec3 objectColor;

const vec2 QUAD[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1),
                            vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    ObjectData object = objects.data[gl_InstanceIndex];
    vec2 corner = QUAD[gl_VertexIndex];
    mat4 viewProjection = mat4(pc.p[0], pc.p[1], pc.p[2], pc.p[3]);
    vec3 position = object.positionScale.xyz + vec3(corner * object.positionScale.w, 0.0);
    gl_Position = viewProjection * vec4(position, 1.0);
    quadUv = corner;
    objectColor = object.color.rgb;
}
