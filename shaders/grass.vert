#version 450
// 第80章：植被渲染 — 草叶顶点着色器
// 每根草从 SSBO 读取位置，展开成四边形
struct GrassBlade {
    vec4 posYaw;   // xyz=根部位置, w=朝向角
    vec4 params;   // x=高度, y=弯曲, z=宽度, w=亮度变化
};

layout(set=0,binding=0) readonly buffer GrassSSBO { GrassBlade blades[]; };
layout(set=0,binding=1) uniform CameraUBO {
    mat4 view, proj;
    vec4 cameraPos;
    vec4 windDir;     // xyz=风向, w=强度
    float time;
    float pad[3];
} ubo;

layout(location=0) out vec3 fragColor;
layout(location=1) out vec2 fragUV;
layout(location=2) out float fragAO;

// 草叶 UV（每根 6 个顶点）
const vec2 BLADE_UV[6] = vec2[](
    vec2(0,0),vec2(1,0),vec2(0.5,1),
    vec2(0,0),vec2(0.5,1),vec2(1,0)  // 双面
);

void main() {
    uint  bladeIdx  = uint(gl_VertexIndex) / 6u;
    uint  vertIdx   = uint(gl_VertexIndex) % 6u;
    GrassBlade b    = blades[bladeIdx];

    vec2  uv        = BLADE_UV[vertIdx];
    float t         = uv.y;   // 0=根部, 1=顶端
    float halfWidth = b.params.z * 0.5 * (1.0-t*0.8);

    // 草叶朝向
    float yaw       = b.posYaw.w;
    vec3  right     = vec3(cos(yaw), 0.0, sin(yaw));
    vec3  up        = vec3(0, 1, 0);

    // 风力偏移（只影响顶端）
    float windPhase = dot(b.posYaw.xyz, ubo.windDir.xyz) * 0.1 + ubo.time * 2.0;
    float windBend  = sin(windPhase) * ubo.windDir.w * t * t;
    vec3  windOff   = ubo.windDir.xyz * windBend;

    // 弯曲曲线
    float bend      = b.params.y * t * t;
    vec3  pos       = b.posYaw.xyz
                    + up * b.params.x * t
                    + right * (uv.x*2.0-1.0) * halfWidth
                    + windOff;

    gl_Position = ubo.proj * ubo.view * vec4(pos, 1.0);

    // 顶端更亮（模拟透光）
    float bright = mix(0.4, 1.0, t) * (0.8 + b.params.w * 0.2);
    fragColor    = vec3(0.25, 0.55, 0.15) * bright;
    fragUV       = uv;
    fragAO       = mix(0.0, 1.0, t);  // 根部 AO 遮蔽
}
