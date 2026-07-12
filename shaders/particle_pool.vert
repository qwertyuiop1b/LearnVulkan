#version 450

struct Particle {
    vec4 position;
    vec4 velocity;
    vec4 color;
    float size;
    float pad[3];
};

layout(std430, set = 0, binding = 0) readonly buffer ParticleBuffer { Particle particles[]; };
layout(std140, set = 0, binding = 1) uniform CameraBlock {
    mat4 view;
    mat4 proj;
    vec3 cameraRight;
    float pad0;
    vec3 cameraUp;
    float pad1;
} camera;
layout(std430, set = 0, binding = 2) readonly buffer AliveIndices { uint aliveIndices[]; };

layout(location = 0) out vec4 fragmentColor;
layout(location = 1) out vec2 fragmentUv;

const vec2 CORNERS[6] = vec2[](
    vec2(-1, 1), vec2(-1, -1), vec2(1, -1),
    vec2(-1, 1), vec2(1, -1), vec2(1, 1));
const vec2 UVS[6] = vec2[](
    vec2(0, 0), vec2(0, 1), vec2(1, 1),
    vec2(0, 0), vec2(1, 1), vec2(1, 0));

void main() {
    uint aliveSlot = uint(gl_VertexIndex) / 6u;
    uint corner = uint(gl_VertexIndex) % 6u;
    Particle particle = particles[aliveIndices[aliveSlot]];
    vec2 offset = CORNERS[corner] * particle.size * 0.5;
    vec3 world = particle.position.xyz + camera.cameraRight * offset.x + camera.cameraUp * offset.y;
    gl_Position = camera.proj * camera.view * vec4(world, 1.0);
    fragmentColor = particle.color;
    fragmentUv = UVS[corner];
}
