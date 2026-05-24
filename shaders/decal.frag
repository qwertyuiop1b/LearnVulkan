#version 450

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D depthTex;

layout(push_constant) uniform DecalPC {
    mat4  decalWorld;
    mat4  decalInvWorld;    // 世界 → 贴花局部空间
    vec4  decalColor;
    vec2  screenSize;
    float pad[2];
} pc;

layout(set = 0, binding = 2) uniform CamInvUBO {
    mat4 invProjView;
} camInv;

void main()
{
    // 当前像素的屏幕 UV
    vec2 screenUV = gl_FragCoord.xy / pc.screenSize;

    // 从深度重建世界坐标
    float depth  = texture(depthTex, screenUV).r;
    if (depth >= 1.0) discard;

    vec4 ndcPos   = vec4(screenUV * 2.0 - 1.0, depth, 1.0);
    vec4 worldH   = camInv.invProjView * ndcPos;
    vec3 worldPos = worldH.xyz / worldH.w;

    // 变换到贴花局部空间
    vec4 local = pc.decalInvWorld * vec4(worldPos, 1.0);

    // box 判断：局部空间 [-0.5, 0.5]^3
    if (any(greaterThan(abs(local.xyz), vec3(0.5)))) discard;

    // 贴花 UV（xz 平面投影）
    vec2 decalUV = local.xz + 0.5;

    // 边缘渐变，避免硬切
    vec2 fade = 1.0 - smoothstep(0.4, 0.5, abs(local.xz));
    float alpha = fade.x * fade.y * pc.decalColor.a;

    outColor = vec4(pc.decalColor.rgb, alpha);
}
