#version 450

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform ShadowPC {
    mat4 lightMVP;
} pc;

void main()
{
    gl_Position = pc.lightMVP * vec4(inPosition, 1.0);
}
