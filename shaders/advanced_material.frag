#version 450
layout(location=0)in vec2 inUV;layout(location=0)out vec4 outColor;layout(set=0,binding=0,std430)readonly buffer Data{vec4 values[];}data;
void main(){uint i=min(uint(inUV.x*1024.),1023u);vec4 v=data.values[i];float a=1.-smoothstep(.008,.018,abs(inUV.y-.17-v.x*.2));float b=1.-smoothstep(.008,.018,abs(inUV.y-.43-v.y*.2));float c=1.-smoothstep(.008,.018,abs(inUV.y-.69-v.z*.2));float d=1.-smoothstep(.008,.018,abs(inUV.y-.9-v.w*.08));outColor=vec4(vec3(.006,.012,.024)+a*vec3(1,.3,.1)+b*vec3(.1,.7,1)+c*vec3(1,.1,.7)+d*vec3(.2,1,.3),1);}
