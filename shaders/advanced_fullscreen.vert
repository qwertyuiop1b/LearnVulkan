#version 450

layout(location = 0) out vec2 outUV;

void main() {
    vec2 position = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    // A positive-height Vulkan viewport maps NDC -Y to the framebuffer top.
    // Keep screen-space +Y aligned with the camera up vector used by ch71-ch83.
    outUV = vec2(position.x * 0.5, 1.0 - position.y * 0.5);
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
