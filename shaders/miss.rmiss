#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;

void main() {
    // 根据光线方向做天空渐变：地平线偏橙，天顶偏蓝
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    float t  = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0); // [-1,1] → [0,1]
    vec3 horizon = vec3(0.8, 0.6, 0.4); // 地平线暖色
    vec3 zenith  = vec3(0.1, 0.3, 0.8); // 天顶蓝色
    hitValue = mix(horizon, zenith, t);
}
