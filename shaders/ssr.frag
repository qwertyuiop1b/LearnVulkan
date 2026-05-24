#version 450
// 屏幕空间反射（第47章）
// 在屏幕空间步进射线；失败时采样 cubemap 作为降级

// 全屏 UV 由 gl_FragCoord 推导

layout(set = 0, binding = 0) uniform sampler2D sceneColor;
layout(set = 0, binding = 1) uniform sampler2D sceneNormal;
layout(set = 0, binding = 2) uniform sampler2D sceneDepth;
layout(set = 0, binding = 3) uniform samplerCube fallbackCubemap;

layout(set = 0, binding = 4) uniform SSRUBO {
    mat4 invView;
    mat4 view;
    mat4 projection;
    vec4 cameraPos;
    float maxDistance;
    float stepSize;
    float thickness;
    float intensity;
} ubo;

layout(location = 0) out vec4 outReflection;

vec3 reconstructViewPos(vec2 uv, float depth)
{
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 view = inverse(ubo.projection) * clip;
    return view.xyz / view.w;
}

bool traceSSR(vec3 viewPos, vec3 viewReflect, out vec2 hitUV)
{
    vec3 rayPos = viewPos;
    vec3 rayDir = normalize(viewReflect);
    float traveled = 0.0;
    for (int i = 0; i < 32; ++i) {
        rayPos += rayDir * ubo.stepSize;
        traveled += ubo.stepSize;
        if (traveled > ubo.maxDistance) break;
        vec4 proj = ubo.projection * vec4(rayPos, 1.0);
        proj.xyz /= proj.w;
        vec2 uv = proj.xy * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;
        float sceneZ = texture(sceneDepth, uv).r;
        vec3 scenePos = reconstructViewPos(uv, sceneZ);
        if (rayPos.z > scenePos.z && rayPos.z - scenePos.z < ubo.thickness) {
            hitUV = uv;
            return true;
        }
    }
    return false;
}

void main()
{
    vec2 vUV = gl_FragCoord.xy / vec2(textureSize(sceneDepth, 0));
    float depth = texture(sceneDepth, vUV).r;
    vec3 viewPos = reconstructViewPos(vUV, depth);
    vec3 normal  = normalize(texture(sceneNormal, vUV).xyz * 2.0 - 1.0);
    vec3 viewN   = mat3(ubo.view) * normal;
    vec3 viewV   = normalize(-viewPos);
    vec3 viewR   = reflect(-viewV, viewN);

    vec2 hitUV;
    vec3 reflColor;
    if (traceSSR(viewPos, viewR, hitUV)) {
        reflColor = texture(sceneColor, hitUV).rgb;
    } else {
        vec3 worldR = mat3(ubo.invView) * viewR;
        reflColor = texture(fallbackCubemap, worldR).rgb;
    }

    float fresnel = pow(1.0 - max(dot(viewV, viewN), 0.0), 3.0);
    outReflection = vec4(reflColor * ubo.intensity, fresnel);
}
