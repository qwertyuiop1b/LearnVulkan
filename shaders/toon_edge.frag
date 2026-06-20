#version 450
/**
 * 第84章（升级版）：法线 + 深度 Sobel 边缘检测
 *
 * 原理：
 *   对法线缓冲区和深度缓冲区各做 3×3 Sobel 算子，
 *   法线不连续（折叠边）或深度不连续（轮廓边）的地方产生线条。
 *   这样可以检测到 Inverted Hull 无法产生的"内部线条"（如面部五官、皱褶）。
 *
 * 效果对比：
 *   Inverted Hull：只有物体外轮廓
 *   Sobel 边缘检测：外轮廓 + 内部结构线 → 更接近真正的手绘风格
 */

layout(location=0) in vec2 inUV;
layout(location=0) out vec4 outColor;

layout(set=0,binding=0) uniform sampler2D sceneTex;   // Pass2 的颜色输出
layout(set=0,binding=1) uniform sampler2D normalTex;  // Pass2 的法线输出
layout(set=0,binding=2) uniform sampler2D depthTex;

layout(push_constant) uniform EdgePC {
    vec2  texelSize;
    float normalThreshold;  // 法线差异阈值（小=更多线条）
    float depthThreshold;   // 深度差异阈值
    float edgeStrength;     // 线条不透明度
    vec4  edgeColor;        // 线条颜色（通常黑色）
    int   onlyEdges;        // 调试：只显示边缘图（不叠加场景）
    float pad[3];
} pc;

// Sobel 算子（3×3）
float sobelNormal(sampler2D tex, vec2 uv) {
    vec2 d = pc.texelSize;
    // Gx 核
    float Gx =
        -1.0 * length(texture(tex, uv + vec2(-d.x,-d.y)).rgb - vec3(0.5)) +
        -2.0 * length(texture(tex, uv + vec2(-d.x, 0.0)).rgb - vec3(0.5)) +
        -1.0 * length(texture(tex, uv + vec2(-d.x, d.y)).rgb - vec3(0.5)) +
         1.0 * length(texture(tex, uv + vec2( d.x,-d.y)).rgb - vec3(0.5)) +
         2.0 * length(texture(tex, uv + vec2( d.x, 0.0)).rgb - vec3(0.5)) +
         1.0 * length(texture(tex, uv + vec2( d.x, d.y)).rgb - vec3(0.5));
    // Gy 核
    float Gy =
        -1.0 * length(texture(tex, uv + vec2(-d.x,-d.y)).rgb - vec3(0.5)) +
        -2.0 * length(texture(tex, uv + vec2( 0.0,-d.y)).rgb - vec3(0.5)) +
        -1.0 * length(texture(tex, uv + vec2( d.x,-d.y)).rgb - vec3(0.5)) +
         1.0 * length(texture(tex, uv + vec2(-d.x, d.y)).rgb - vec3(0.5)) +
         2.0 * length(texture(tex, uv + vec2( 0.0, d.y)).rgb - vec3(0.5)) +
         1.0 * length(texture(tex, uv + vec2( d.x, d.y)).rgb - vec3(0.5));
    return sqrt(Gx*Gx + Gy*Gy);
}

float sobelDepth(sampler2D tex, vec2 uv) {
    vec2 d = pc.texelSize;
    float tl = texture(tex, uv + vec2(-d.x,-d.y)).r;
    float t  = texture(tex, uv + vec2( 0.0,-d.y)).r;
    float tr = texture(tex, uv + vec2( d.x,-d.y)).r;
    float ml = texture(tex, uv + vec2(-d.x, 0.0)).r;
    float mr = texture(tex, uv + vec2( d.x, 0.0)).r;
    float bl = texture(tex, uv + vec2(-d.x, d.y)).r;
    float b  = texture(tex, uv + vec2( 0.0, d.y)).r;
    float br = texture(tex, uv + vec2( d.x, d.y)).r;

    float Gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
    float Gy = -tl - 2.0*t  - tr + bl + 2.0*b  + br;
    return sqrt(Gx*Gx + Gy*Gy);
}

void main()
{
    vec3 sceneColor = texture(sceneTex, inUV).rgb;

    // 法线边缘（内部折叠边）
    float normEdge  = sobelNormal(normalTex, inUV);
    float normalEdge= smoothstep(pc.normalThreshold, pc.normalThreshold * 2.0, normEdge);

    // 深度边缘（轮廓边 / 物体边界）
    float depthEdge = sobelDepth(depthTex, inUV);
    depthEdge = smoothstep(pc.depthThreshold, pc.depthThreshold * 2.0, depthEdge);

    // 合并两种边缘
    float edge = clamp(max(normalEdge, depthEdge), 0.0, 1.0);
    edge      *= pc.edgeStrength;

    vec3 finalColor;
    if (pc.onlyEdges != 0) {
        // 调试模式：白底黑线
        finalColor = mix(vec3(1.0), pc.edgeColor.rgb, edge);
    } else {
        // 正常模式：叠加边缘线到场景
        finalColor = mix(sceneColor, pc.edgeColor.rgb, edge * pc.edgeColor.a);
    }

    outColor = vec4(finalColor, 1.0);
}
