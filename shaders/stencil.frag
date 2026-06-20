#version 450

layout(location = 0) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    fragColor = inColor;
}
