#version 450
layout(location=0) in vec2 inUV; layout(location=0) out vec4 outColor;
layout(set=0,binding=0,std430) readonly buffer Data { vec4 values[]; } data;
void main(){ uint i=min(uint(inUV.x*1024.),1023u); vec4 v=data.values[i]; float l=1.-smoothstep(.008,.018,abs(inUV.y-.16-v.x*.22)); float h=1.-smoothstep(.008,.018,abs(inUV.y-.42-v.y*.22)); float r=1.-smoothstep(.008,.018,abs(inUV.y-.68-v.z*.22)); float s=1.-smoothstep(.008,.018,abs(inUV.y-.9-v.w*.08)); outColor=vec4(vec3(.006,.012,.024)+l*vec3(.15,.4,1)+h*vec3(1,.3,.06)+r*vec3(.1,1,.4)+s*vec3(1,.1,1),1); }
