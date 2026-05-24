#version 450
// SSAO 核心计算（第46章）
// 读取 G-Buffer 位置/法线，在半球内采样估计环境光遮蔽

// 全屏四边形由 deferred_lighting.vert 生成，UV 由 gl_FragCoord 推导

layout(set = 0, binding = 0) uniform sampler2D gPosition;
layout(set = 0, binding = 1) uniform sampler2D gNormal;
layout(set = 0, binding = 2) uniform sampler2D noiseTex;

layout(set = 0, binding = 3) uniform SSAOUBO {
    mat4 projection;
    mat4 invProjection;
    vec4 samples[32];
    vec2 noiseScale;
    float radius;
    float bias;
} ubo;

layout(location = 0) out float outAO;

const int KERNEL_SIZE = 32;

void main()
{
    vec2 vUV = gl_FragCoord.xy / vec2(textureSize(gPosition, 0));
    vec3 fragPos = texture(gPosition, vUV).xyz;
    vec3 normal  = normalize(texture(gNormal, vUV).xyz * 2.0 - 1.0);
    vec3 randomVec = normalize(texture(noiseTex, vUV * ubo.noiseScale).xyz * 2.0 - 1.0);

    vec3 tangent   = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        vec3 samplePos = TBN * ubo.samples[i].xyz;
        samplePos    = fragPos + samplePos * ubo.radius;
        vec4 offset    = ubo.projection * vec4(samplePos, 1.0);
        offset.xyz    /= offset.w;
        offset.xyz     = offset.xyz * 0.5 + 0.5;
        if (offset.x < 0.0 || offset.x > 1.0 || offset.y < 0.0 || offset.y > 1.0)
            continue;
        float sampleDepth = texture(gPosition, offset.xy).z;
        float rangeCheck  = smoothstep(0.0, 1.0, ubo.radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth >= samplePos.z + ubo.bias ? 1.0 : 0.0) * rangeCheck;
    }
    outAO = 1.0 - (occlusion / float(KERNEL_SIZE));
}
