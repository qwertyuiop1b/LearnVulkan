#version 450
// G-Buffer 几何通道片段着色器（第19章）
// 将各种几何属性写入多个 MRT（Multiple Render Targets）附件
//
// G-Buffer 布局：
//   attachment 0: RGB = 世界位置 (Position.xyz)
//   attachment 1: RGB = 法线向量 (Normal.xyz)
//   attachment 2: RGB = 漫反射颜色 (Albedo.rgb)

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;

// 多渲染目标输出（MRT：Multiple Render Targets）
// 每个 layout(location=N) 对应 Framebuffer 的第 N 个颜色附件
layout(location = 0) out vec4 outPosition;   // 世界坐标
layout(location = 1) out vec4 outNormal;     // 法线（[-1,1] → [0,1]）
layout(location = 2) out vec4 outAlbedo;     // 漫反射颜色

void main()
{
    outPosition = vec4(fragPos, 1.0);
    // 法线从 [-1,1] 编码到 [0,1]（方便存储为颜色格式）
    outNormal   = vec4(fragNormal * 0.5 + 0.5, 1.0);
    outAlbedo   = vec4(fragColor, 1.0);
}
