#version 450
// 第74章：SSGI（屏幕空间全局光照）
// 与 SSAO 的区别：采样周围法线+颜色，而不只是遮挡
layout(location=0) in vec2 inUV;
layout(location=0) out vec4 outGI;  // rgb=间接光颜色

layout(set=0,binding=0) uniform sampler2D posTex;      // world pos
layout(set=0,binding=1) uniform sampler2D normalTex;   // world normal
layout(set=0,binding=2) uniform sampler2D albedoTex;   // albedo
layout(set=0,binding=3) uniform sampler2D depthTex;
layout(set=0,binding=4) uniform sampler2D noiseTex;    // 随机旋转噪声

layout(push_constant) uniform SSGIPC {
    mat4  proj;
    mat4  view;
    vec4  cameraPos;
    float sampleRadius;   // 采样半径（世界空间）
    float intensity;
    float maxDistance;    // 最大采样距离
    int   numSamples;
} pc;

// 半球方向（余弦分布）
vec3 cosineHemisphere(vec2 xi, vec3 N) {
    float phi   = xi.x * 6.28318;
    float theta = acos(sqrt(xi.y));
    float sinT  = sin(theta);
    vec3 H      = vec3(sinT*cos(phi), cos(theta), sinT*sin(phi));
    vec3 up     = abs(N.y) < 0.99 ? vec3(0,1,0) : vec3(1,0,0);
    vec3 T      = normalize(cross(up, N));
    vec3 B      = cross(N, T);
    return normalize(T*H.x + N*H.y + B*H.z);
}

void main() {
    vec3 pos    = texture(posTex,    inUV).xyz;
    vec3 normal = normalize(texture(normalTex, inUV).xyz);
    vec3 albedo = texture(albedoTex, inUV).rgb;
    float depth = texture(depthTex,  inUV).r;

    if (depth >= 1.0) { outGI = vec4(0.0); return; }

    vec2 noiseScale = vec2(textureSize(posTex,0)) / 4.0;
    vec3 noise      = texture(noiseTex, inUV*noiseScale).xyz * 2.0 - 1.0;

    vec3 gi     = vec3(0.0);
    float total = 0.0;

    for (int i = 0; i < min(pc.numSamples, 16); ++i) {
        // 余弦分布采样方向
        float fi  = float(i) + 0.5;
        float phi = fi * 2.399963;    // 黄金角 ~137.5°
        float r   = sqrt(fi / float(pc.numSamples));
        vec2  xi  = fract(vec2(r, phi/6.28318) + noise.xy);
        vec3  sampleDir = cosineHemisphere(xi, normal);

        // 屏幕空间步进
        vec3  samplePos = pos + sampleDir * pc.sampleRadius;
        vec4  sampleClip = pc.proj * pc.view * vec4(samplePos, 1.0);
        vec2  sampleUV   = sampleClip.xy / sampleClip.w * 0.5 + 0.5;

        if (any(lessThan(sampleUV, vec2(0))) || any(greaterThan(sampleUV, vec2(1)))) continue;

        vec3  sampleAlbedo = texture(albedoTex, sampleUV).rgb;
        vec3  sampleNormal = normalize(texture(normalTex, sampleUV).xyz);

        // 朗伯反射：样点法线指向当前点方向
        float NdotL = max(dot(normal, sampleDir), 0.0);
        float NdotL2= max(-dot(sampleNormal, sampleDir), 0.0);
        float contrib = NdotL * NdotL2;

        gi    += sampleAlbedo * contrib;
        total += 1.0;
    }

    if (total > 0.0) gi /= total;
    outGI = vec4(gi * pc.intensity, 1.0);
}
