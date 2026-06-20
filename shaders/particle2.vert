#version 450

// Billboard 粒子：每个粒子 6 个顶点（两个三角形），由 SSBO 展开
// Draw(N_PARTICLES * 6, 1, 0, 0) → 无需顶点缓冲区

struct Particle {
    vec4 position;   // xyz=pos, w=lifetime
    vec4 velocity;
    vec4 color;
    float size;
    float pad[3];
};

layout(set = 0, binding = 0) readonly buffer ParticleSSBO {
    Particle particles[];
};

layout(set = 0, binding = 1) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec3 cameraRight;
    float pad0;
    vec3 cameraUp;
    float pad1;
} cam;

// 四边形的 6 个顶点偏移（两个三角形，CCW）
const vec2 QUAD[6] = vec2[](
    vec2(-1.0,  1.0), vec2(-1.0, -1.0), vec2( 1.0, -1.0),
    vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0)
);
const vec2 UV[6] = vec2[](
    vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0),
    vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(1.0, 0.0)
);

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragUV;

void main()
{
    uint particleIdx = uint(gl_VertexIndex) / 6u;
    uint cornerIdx   = uint(gl_VertexIndex) % 6u;

    Particle p = particles[particleIdx];

    // 死亡粒子移到裁剪空间之外，不渲染
    if (p.position.w <= 0.0) {
        gl_Position = vec4(10.0, 10.0, 10.0, 1.0);
        fragColor   = vec4(0.0);
        fragUV      = vec2(0.0);
        return;
    }

    vec2 corner  = QUAD[cornerIdx];
    vec3 worldPos = p.position.xyz
                  + cam.cameraRight * corner.x * p.size * 0.5
                  + cam.cameraUp    * corner.y * p.size * 0.5;

    gl_Position = cam.proj * cam.view * vec4(worldPos, 1.0);
    fragColor   = p.color;
    fragUV      = UV[cornerIdx];
}
