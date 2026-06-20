#version 450
// GPU 蒙皮：骨骼矩阵 UBO 混合顶点

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in uvec4 inJoints;
layout(location = 4) in vec4 inWeights;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 projection;
} scene;

layout(set = 0, binding = 1) uniform BoneUBO {
    mat4 bones[64];
} boneData;

void main() {
    mat4 skinMatrix = mat4(0.0);
    skinMatrix += boneData.bones[inJoints.x] * inWeights.x;
    skinMatrix += boneData.bones[inJoints.y] * inWeights.y;
    skinMatrix += boneData.bones[inJoints.z] * inWeights.z;
    skinMatrix += boneData.bones[inJoints.w] * inWeights.w;
    float weightSum = inWeights.x + inWeights.y + inWeights.z + inWeights.w;
    if (weightSum < 1e-5)
        skinMatrix = boneData.bones[0];
    vec4 skinnedPos = skinMatrix * vec4(inPosition, 1.0);
    vec3 skinnedNormal = mat3(skinMatrix) * inNormal;
    gl_Position = scene.projection * scene.view * skinnedPos;
    fragNormal = normalize(skinnedNormal);
    fragTexCoord = inTexCoord;
}
