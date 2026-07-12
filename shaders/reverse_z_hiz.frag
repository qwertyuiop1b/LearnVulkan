#version 450
layout(location=0) in vec2 inUV; layout(location=0) out vec4 outColor;
layout(set=0,binding=0,std430) readonly buffer Data { vec4 values[]; } data;
layout(push_constant) uniform Push { vec4 p[8]; } pc;
void main(){ uint i=min(uint(inUV.x*1024.0),1023u); vec4 v=data.values[i]; float line=1.0-smoothstep(.008,.02,abs(inUV.y-v.y)); float pyramid=1.0-smoothstep(.008,.02,abs(inUV.y-v.z)); vec3 c=vec3(.01,.02,.04)+line*vec3(.15,.8,1.0)+pyramid*vec3(1.0,.28,.06); c+=vec3(.1,.2,.35)*smoothstep(0.,1.,v.w)*.2; outColor=vec4(c,1); }
