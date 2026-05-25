#version 450
// 第73章：大气散射（Rayleigh + Mie 单次散射近似）
layout(location=0) in vec2 inUV;
layout(location=0) out vec4 outColor;

layout(push_constant) uniform AtmoPC {
    vec4 sunDir;         // xyz=方向（归一化，指向太阳）, w=强度
    vec4 cameraPos;
    float earthRadius;   // 6371000 m
    float atmoRadius;    // 6471000 m（大气层顶）
    float Hr;            // Rayleigh 标高 8500m
    float Hm;            // Mie 标高 1200m
} pc;

// 从 UV 和深度重建方向向量
layout(set=0,binding=0) uniform sampler2D depthTex;

vec3 uvToDir(vec2 uv) {
    // 等经纬映射（全景天空）
    float phi   = uv.x * 6.28318 - 3.14159;
    float theta = uv.y * 3.14159;
    return normalize(vec3(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi)));
}

// Henyey-Greenstein 相位函数（Mie 散射）
float hgPhase(float cosTheta, float g) {
    float gg = g*g;
    return (1.0-gg) / (4.0*3.14159*pow(1.0+gg-2.0*g*cosTheta, 1.5));
}

// Rayleigh 相位函数
float rayleighPhase(float cosTheta) {
    return 0.75 * (1.0 + cosTheta*cosTheta);
}

void main() {
    vec3 dir     = uvToDir(inUV);
    vec3 sunDir  = normalize(pc.sunDir.xyz);
    float cosAngle = dot(dir, sunDir);

    // 简化版：单次散射（不进行光线步进，用解析近似）
    float altitude = pc.cameraPos.y;   // 相机高度（米）
    float height   = max(altitude, 0.0);

    // Rayleigh 散射系数（λ=450nm blue → 最强）
    vec3 betaR = vec3(5.8e-6, 13.5e-6, 33.1e-6);
    // Mie 散射（灰色，方向性强）
    float betaM = 21e-6;
    float g     = 0.76;   // 不对称参数

    // 光学深度近似（垂直方向指数衰减）
    float optDepthR = exp(-height/pc.Hr);
    float optDepthM = exp(-height/pc.Hm);

    float phR = rayleighPhase(cosAngle);
    float phM = hgPhase(cosAngle, g);

    vec3 inscatter = betaR * optDepthR * phR + vec3(betaM * optDepthM * phM);
    inscatter     *= pc.sunDir.w;

    // 地平线渐变（dir.y < 0 快速变暗）
    float horizon  = smoothstep(-0.05, 0.1, dir.y);
    inscatter     *= horizon;

    // 太阳圆盘
    float sunDisk  = smoothstep(0.9995, 0.9999, cosAngle);
    vec3  sunColor = vec3(1.0, 0.95, 0.8) * pc.sunDir.w * 10.0 * sunDisk;

    outColor = vec4(inscatter + sunColor, 1.0);
}
