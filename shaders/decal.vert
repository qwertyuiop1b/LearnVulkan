#version 450

// 贴花：渲染一个 box 体积，片元着色器中重建世界坐标判断是否在 box 内

layout(location = 0) in vec3 inPosition;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
} cam;

layout(push_constant) uniform DecalPC {
    mat4  decalWorld;       // box 的 world 矩阵（中心 + 旋转 + 缩放）
    mat4  decalInvWorld;    // 逆矩阵，用于将世界坐标变换到贴花局部空间
    vec4  decalColor;
    vec2  screenSize;
    float pad[2];
} pc;

void main()
{
    gl_Position = cam.proj * cam.view * pc.decalWorld * vec4(inPosition, 1.0);
}
