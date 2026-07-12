#version 450

struct SurfaceVertex {
    vec4 position;
    vec4 normal;
};

layout(std430, set = 0, binding = 0) readonly buffer SurfaceBuffer {
    SurfaceVertex vertices[];
} surface;

layout(push_constant) uniform CameraParams {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 renderParams; // time, exposure, rim strength, reserved
} pc;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out float materialNoise;

void main() {
    SurfaceVertex vertex = surface.vertices[uint(gl_VertexIndex)];
    worldPosition = vertex.position.xyz;
    worldNormal = vertex.normal.xyz;
    materialNoise = vertex.position.w;
    gl_Position = pc.viewProjection * vec4(worldPosition, 1.0);
}
