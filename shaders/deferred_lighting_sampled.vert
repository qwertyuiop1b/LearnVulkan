#version 450

layout(location = 0) out vec2 fragUV;

void main()
{
    vec2 position = vec2((gl_VertexIndex & 1) << 2, (gl_VertexIndex & 2) << 1) - 1.0;
    gl_Position = vec4(position, 0.0, 1.0);
    fragUV = position * 0.5 + 0.5;
}
