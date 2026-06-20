#version 450
// 第75章：LUT 色彩分级（3D Look-Up Table）
layout(location=0) in vec2 inUV;
layout(location=0) out vec4 outColor;

layout(set=0,binding=0) uniform sampler2D sceneTex;
layout(set=0,binding=1) uniform sampler3D lutTex;     // 3D LUT（16x16x16 或 32x32x32）

layout(push_constant) uniform LutPC {
    float lutSize;     // LUT 每轴分辨率（16 or 32）
    float blendAmount; // 0=原始, 1=完全 LUT
    int   presetIdx;   // 0=无,1=电影,2=夜视,3=暖调,4=冷调
    float pad;
} pc;

// 将 [0,1] 颜色转换为 LUT 索引（带半像素偏移）
vec3 toLutCoord(vec3 color) {
    float scale = (pc.lutSize - 1.0) / pc.lutSize;
    float offset= 0.5 / pc.lutSize;
    return color * scale + offset;
}

void main() {
    vec3 hdrColor = texture(sceneTex, inUV).rgb;

    // ACES Tone Map 先（让颜色在 [0,1]）
    vec3 mapped  = hdrColor / (hdrColor + 1.0);

    // 采样 3D LUT
    vec3 lutColor= texture(lutTex, toLutCoord(mapped)).rgb;

    // 混合原始 + LUT
    vec3 result  = mix(mapped, lutColor, pc.blendAmount);

    outColor = vec4(result, 1.0);
}
