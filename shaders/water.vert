#version 450

layout(location = 0) in vec3 inPosition;

layout(binding = 0) uniform WaterUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 reflectView;   // 反射相机的 view 矩阵
    vec4 cameraPos;
    float time;
    float waveHeight;
    float waveSpeed;
    float tiling;
} ubo;

layout(location = 0) out vec4 clipSpacePos;    // 用于反射/折射采样
layout(location = 1) out vec3 fragPos;
layout(location = 2) out vec3 toCamera;
layout(location = 3) out vec2 waterUV;
layout(location = 4) out float time;

// Gerstner 波近似（顶点位移）
vec3 gerstnerWave(vec3 pos, float amp, vec2 dir, float speed, float waveLen)
{
    float k     = 6.28318 / waveLen;
    float phase = dot(pos.xz, dir) * k + ubo.time * speed;
    pos.y += amp * sin(phase);
    return pos;
}

void main()
{
    vec3 pos = inPosition;
    // 叠加两道 Gerstner 波
    pos = gerstnerWave(pos, ubo.waveHeight,        vec2(1.0, 0.3), ubo.waveSpeed,       8.0);
    pos = gerstnerWave(pos, ubo.waveHeight * 0.6,  vec2(-0.5, 1.0), ubo.waveSpeed * 1.3, 5.0);

    vec4 worldPos = ubo.model * vec4(pos, 1.0);
    gl_Position   = ubo.proj * ubo.view * worldPos;
    clipSpacePos  = gl_Position;
    fragPos       = worldPos.xyz;
    toCamera      = ubo.cameraPos.xyz - worldPos.xyz;
    waterUV       = inPosition.xz * ubo.tiling;
    time          = ubo.time;
}
