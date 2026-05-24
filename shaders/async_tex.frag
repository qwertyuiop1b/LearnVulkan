#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(push_constant) uniform QuadPC {
    vec2 pos;
    vec2 size;
    float loadProgress;  // 0=加载中, 1=完成（用于显示进度遮罩）
    float pad[3];
} pc;

void main()
{
    vec4 texColor = texture(tex, inUV);

    // 加载进度遮罩：从左到右扫描线显示
    if (inUV.x > pc.loadProgress) {
        // 未加载区域：显示棋盘格占位
        ivec2 checker = ivec2(inUV * 8.0);
        float c = float((checker.x + checker.y) % 2) * 0.3 + 0.2;
        texColor = vec4(c, c, c, 1.0);
    }

    outColor = texColor;
}
