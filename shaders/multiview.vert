#version 450
// Multiview 顶点着色器（第35章）
// gl_ViewIndex：当前视图（0=左眼，1=右眼）
//
// 一次绘制调用同时渲染左眼和右眼！

#extension GL_EXT_multiview : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(set = 0, binding = 0) uniform UBO {
    mat4 viewLeft;
    mat4 viewRight;
    mat4 projection;
} ubo;

layout(location = 0) out vec3 fragColor;

void main()
{
    // 根据当前视图（左/右眼）选择不同的 view 矩阵
    mat4 view = (gl_ViewIndex == 0) ? ubo.viewLeft : ubo.viewRight;
    gl_Position = ubo.projection * view * vec4(inPosition, 1.0);
    fragColor   = inColor;

    // 轻微着色差异：左眼偏红，右眼偏蓝（便于视觉验证）
    if (gl_ViewIndex == 0)
        fragColor = mix(fragColor, vec3(1,0.5,0.5), 0.15);
    else
        fragColor = mix(fragColor, vec3(0.5,0.5,1), 0.15);
}
