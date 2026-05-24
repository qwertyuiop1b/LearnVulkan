#version 450

layout(location = 0) out vec2 outUV;

layout(push_constant) uniform QuadPC {
    vec2 pos;    // NDC 左下角
    vec2 size;   // NDC 宽高
} pc;

const vec2 CORNERS[6] = vec2[](
    vec2(0,0), vec2(1,0), vec2(1,1),
    vec2(0,0), vec2(1,1), vec2(0,1)
);

void main()
{
    vec2 uv  = CORNERS[gl_VertexIndex];
    outUV    = uv;
    gl_Position = vec4(pc.pos + uv * pc.size, 0.0, 1.0);
}
