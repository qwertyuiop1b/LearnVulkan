#version 450
// Push Descriptors 演示顶点着色器（第33章）
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec3 inColor;

// Push Descriptor: 整个帧只 push 一次，无需预先分配描述符池
layout(set = 0, binding = 0) uniform UBO { mat4 mvp; float time; } ubo;

layout(location = 0) out vec3 fragColor;

void main()
{
    float angle = ubo.time + float(gl_InstanceIndex) * 0.4;
    vec2 pos = vec2(
        inPos.x * cos(angle) - inPos.y * sin(angle),
        inPos.x * sin(angle) + inPos.y * cos(angle)
    );
    pos += vec2(cos(float(gl_InstanceIndex) * 1.3), sin(float(gl_InstanceIndex) * 1.7)) * 0.7;
    gl_Position = ubo.mvp * vec4(pos, 0.0, 1.0);
    fragColor   = inColor * (0.5 + 0.5 * sin(ubo.time + float(gl_InstanceIndex)));
}
