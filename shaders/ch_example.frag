#version 450

layout(location = 0)in vec3 fragColor;

layout(location = 0)out vec4 outClolor;

void main() {
    outClolor = vec4(fragColor, 1.0f);
}