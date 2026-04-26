#version 450
// G-Buffer 几何通道顶点着色器（第19章）
// 将场景几何信息写入多个 G-Buffer 附件

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(set = 0, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragTexCoord;

void main()
{
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    gl_Position   = ubo.projection * ubo.view * worldPos;
    fragPos       = worldPos.xyz;
    fragNormal    = normalize(mat3(transpose(inverse(ubo.model))) * inNormal);
    fragColor     = inColor;
    fragTexCoord  = inTexCoord;
}
