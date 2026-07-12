#version 450

layout(push_constant) uniform AttackPush {
    float time;
    float unusedX;
    float unusedY;
    float viewportWidth;
    float viewportHeight;
} pc;

layout(location = 0) out vec2 ribbonUv;
layout(location = 1) out vec3 worldPosition;
layout(location = 2) out vec3 worldNormal;
layout(location = 3) flat out uint ribbonId;

const int SEGMENTS = 64;

vec3 bezier(vec3 a, vec3 b, vec3 c, float t) {
    float u = 1.0 - t;
    return u * u * a + 2.0 * u * t * b + t * t * c;
}

vec3 bezierTangent(vec3 a, vec3 b, vec3 c, float t) {
    return normalize(2.0 * (1.0 - t) * (b - a) + 2.0 * t * (c - b));
}

mat3 rotationY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
}

mat3 rotationX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);
}

void main() {
    int verticesPerRibbon = SEGMENTS * 6;
    int localVertex = gl_VertexIndex % verticesPerRibbon;
    ribbonId = uint(gl_VertexIndex / verticesPerRibbon);
    int segment = localVertex / 6;
    int corner = localVertex % 6;

    // Triangle list corners: (t0,-), (t0,+), (t1,+), (t0,-), (t1,+), (t1,-).
    const int endpointLut[6] = int[](0, 0, 1, 0, 1, 1);
    const float sideLut[6] = float[](-1.0, 1.0, 1.0, -1.0, 1.0, -1.0);
    float t = float(segment + endpointLut[corner]) / float(SEGMENTS);
    float sideSign = sideLut[corner];

    vec3 a;
    vec3 b;
    vec3 c;
    float baseWidth;
    if (ribbonId == 0u) {
        // Main slash travels toward the camera before bending away at the tip.
        a = vec3(-1.38, -0.72, -0.38);
        b = vec3( 0.02,  1.02,  0.72);
        c = vec3( 1.34,  0.22, -0.18);
        baseWidth = mix(0.28, 0.035, smoothstep(0.02, 0.96, t));
    } else {
        // Counter slash is behind the main ribbon and crosses at another angle.
        a = vec3(-1.45, 0.34, -0.62);
        b = vec3(-0.72, 1.12, -0.20);
        c = vec3( 0.12, 0.18,  0.30);
        baseWidth = mix(0.15, 0.018, smoothstep(0.0, 0.94, t));
    }

    vec3 center = bezier(a, b, c, t);
    vec3 tangent = bezierTangent(a, b, c, t);
    vec3 viewDirection = normalize(vec3(0.0, 0.0, 3.4) - center);
    vec3 side = normalize(cross(tangent, viewDirection));

    // Twist the ribbon around its tangent. This turns a flat streak into a 3D surface.
    float twist = sin(t * 5.2 + float(ribbonId) * 1.7) * 0.72;
    vec3 faceNormal = normalize(cross(tangent, side));
    vec3 twistedSide = normalize(side * cos(twist) + faceNormal * sin(twist));
    vec3 position = center + twistedSide * baseWidth * sideSign;

    // Fast anticipation and follow-through provide acceleration instead of uniform motion.
    float cycle = fract(pc.time / 3.0);
    float attack = smoothstep(0.02, 0.42, cycle);
    float swingAngle = mix(-0.48, 0.20, attack);
    mat3 swing = rotationY(swingAngle) * rotationX(-0.10 + 0.16 * attack);
    position = swing * position;
    tangent = swing * tangent;
    twistedSide = swing * twistedSide;
    faceNormal = normalize(cross(tangent, twistedSide));

    float aspect = max(pc.viewportWidth / max(pc.viewportHeight, 1.0), 0.1);
    float cameraDistance = 3.35 - position.z;
    float focalLength = 1.72;
    float nearPlane = 0.1;
    float farPlane = 10.0;
    float clipZ = ((cameraDistance - nearPlane) / (farPlane - nearPlane)) * cameraDistance;
    gl_Position = vec4(position.x * focalLength / aspect,
                       position.y * focalLength,
                       clipZ,
                       cameraDistance);

    ribbonUv = vec2(t, sideSign * 0.5 + 0.5);
    worldPosition = position;
    worldNormal = faceNormal;
}
