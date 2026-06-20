#version 450
// 天空盒顶点着色器：移除 view 平移，深度固定为 1.0

layout(location = 0) in vec3 inPosition;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 projection;
    mat4 view;
} ubo;

layout(location = 0) out vec3 fragTexCoord;

void main()
{
    fragTexCoord = inPosition;
    mat4 rotView = mat4(mat3(ubo.view));
    vec4 pos = ubo.projection * rotView * vec4(inPosition, 1.0);
    gl_Position = pos.xyww;
}
