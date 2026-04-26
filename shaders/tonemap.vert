#version 450
// Post-Processing: 最终合成 + Tone Mapping + Gamma Correction

// 全屏三角形（顶点在着色器中生成，无顶点缓冲）
void main()
{
    vec2 pos = vec2((gl_VertexIndex & 1) << 2, (gl_VertexIndex & 2) << 1) - 1.0;
    gl_Position = vec4(pos, 0.0, 1.0);
}
