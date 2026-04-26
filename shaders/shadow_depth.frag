#version 450
// 阴影深度通道片段着色器（第18章）
// 深度值由光栅化自动写入深度缓冲，无需输出颜色

void main()
{
    // gl_FragDepth 自动写入，此着色器可以为空
    // （可在此添加 alpha test 等逻辑）
}
