#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(push_constant) uniform CapturePC {
    mat4 viewProj;  // 探针某一面的 viewProj（90° FOV）
} pc;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

void main()
{
    fragPos    = inPosition;
    fragNormal = inNormal;
    fragColor  = inColor;
    gl_Position = pc.viewProj * vec4(inPosition, 1.0);
}
