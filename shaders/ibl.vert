#version 450
// IBL 演示：UV 球体顶点着色器

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 cameraPos;
} ubo;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;

void main()
{
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;
    mat3 normalMatrix = mat3(transpose(inverse(ubo.model)));
    fragNormal = normalize(normalMatrix * inNormal);
    fragTexCoord = inTexCoord;
    gl_Position = ubo.projection * ubo.view * worldPos;
}
