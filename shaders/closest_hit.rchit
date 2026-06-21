#version 460
#extension GL_EXT_ray_tracing : require

struct HitPayload {
    vec3  attenuation;
    vec3  nextOrigin;
    vec3  nextDir;
    bool  scatter;
};
layout(location = 0) rayPayloadInEXT HitPayload payload;

// 材质 SSBO（与 C++ MaterialGPU 对齐）
struct Material {
    vec4  albedo;   // xyz=albedo, w=fuzz(Metal)/ior(Dielectric)
    int   type;     // 0=Lambert, 1=Metal, 2=Dielectric
    int   pad0, pad1, pad2;
};
layout(set = 0, binding = 2) readonly buffer MatBuf { Material m[]; } mats;

// ── 随机数（与 rgen 相同的 PCG） ──────────────────────────────────────────────
uint rng;
uint pcg() {
    rng = rng * 747796405u + 2891336453u;
    uint w = ((rng >> ((rng >> 28u) + 4u)) ^ rng) * 277803737u;
    return (w >> 22u) ^ w;
}
float rand01() { return float(pcg()) / 4294967295.0; }
float rand11() { return rand01() * 2.0 - 1.0; }

vec3 randInUnitSphere() {
    vec3 p;
    for (int i = 0; i < 8; ++i) {
        p = vec3(rand11(), rand11(), rand11());
        if (dot(p, p) < 1.0) return p;
    }
    return p;
}

vec3 randUnitVec() { return normalize(randInUnitSphere()); }

// Schlick 菲涅耳近似
float schlick(float cosine, float ior) {
    float r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

void main() {
    // ── 播种随机数：实例 + 图元 + 射线方向哈希 ──────────────────────────
    rng = uint(gl_InstanceCustomIndexEXT) * 1000003u
        + uint(gl_PrimitiveID) * 7u
        + uint(gl_HitTEXT * 1e4);
    pcg(); pcg();

    // ── 计算世界空间法线（单位球体：物体坐标 = 法线） ─────────────────
    vec3 objHit    = gl_ObjectRayOriginEXT + gl_HitTEXT * gl_ObjectRayDirectionEXT;
    vec3 objNormal = normalize(objHit);
    // gl_ObjectToWorldEXT 是 mat4x3（4列3行），提取 3×3 旋转缩放部分
    mat3 o2w       = mat3(gl_ObjectToWorldEXT[0].xyz,
                          gl_ObjectToWorldEXT[1].xyz,
                          gl_ObjectToWorldEXT[2].xyz);
    vec3 N = normalize(o2w * objNormal);
    vec3 P = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 V = gl_WorldRayDirectionEXT; // 入射方向（已归一化）

    // 确保法线朝向入射一侧
    bool frontFace = dot(V, N) < 0.0;
    if (!frontFace) N = -N;

    Material mat = mats.m[gl_InstanceCustomIndexEXT];

    if (mat.type == 0) {
        // ── Lambertian 漫反射 ─────────────────────────────────────────
        vec3 scatter = normalize(N + randUnitVec());
        if (dot(scatter, scatter) < 1e-6) scatter = N; // 退化保护
        payload.attenuation = mat.albedo.xyz;
        payload.nextOrigin  = P + 1e-4 * N;
        payload.nextDir     = scatter;
        payload.scatter     = true;

    } else if (mat.type == 1) {
        // ── Metal 镜面反射 ────────────────────────────────────────────
        float fuzz    = mat.albedo.w;
        vec3 reflected = reflect(V, N) + fuzz * randInUnitSphere();
        payload.attenuation = mat.albedo.xyz;
        payload.nextOrigin  = P + 1e-4 * N;
        payload.nextDir     = normalize(reflected);
        payload.scatter     = dot(payload.nextDir, N) > 0.0;

    } else {
        // ── Dielectric 玻璃折射 ───────────────────────────────────────
        float ior         = mat.albedo.w;
        float etaRatio    = frontFace ? (1.0 / ior) : ior;
        float cosTheta    = min(dot(-V, N), 1.0);
        float sinTheta    = sqrt(1.0 - cosTheta * cosTheta);
        bool  canRefract  = etaRatio * sinTheta <= 1.0;
        vec3  dir;
        if (canRefract && schlick(cosTheta, etaRatio) < rand01())
            dir = refract(V, N, etaRatio);
        else
            dir = reflect(V, N);

        payload.attenuation = vec3(1.0);
        payload.nextOrigin  = P + 1e-4 * (dot(dir, N) > 0.0 ? N : -N);
        payload.nextDir     = dir;
        payload.scatter     = true;
    }
}
