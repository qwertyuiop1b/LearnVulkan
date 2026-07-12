#version 450

struct Body { vec4 positionRadius; vec4 velocityRestitution; };
layout(set = 0, binding = 0, std430) readonly buffer Bodies { Body bodies[]; };
layout(push_constant) uniform Push { vec4 p[8]; } pc;
layout(location = 0) out vec2 vLocal;
layout(location = 1) out vec3 vColor;

void main() {
    const vec2 corners[6] = vec2[6](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,-1), vec2(1,1), vec2(-1,1));
    Body b = bodies[gl_InstanceIndex];
    vec2 corner = corners[gl_VertexIndex];
    vec3 toEye = normalize(pc.p[4].xyz - b.positionRadius.xyz);
    vec3 right = normalize(cross(vec3(0,1,0), toEye));
    vec3 up = normalize(cross(toEye, right));
    vec3 world = b.positionRadius.xyz + (right * corner.x + up * corner.y) * b.positionRadius.w;
    gl_Position = mat4(pc.p[0], pc.p[1], pc.p[2], pc.p[3]) * vec4(world, 1.0);
    vLocal = corner;
    float speed = length(b.velocityRestitution.xyz);
    vColor = mix(vec3(0.08, 0.45, 1.0), vec3(1.0, 0.22, 0.06), clamp(speed * 0.16, 0.0, 1.0));
}
